import argparse, glob, os, shutil, sys
from datetime import datetime

import yaml

from MSA.script.colab_msa_template_search.colab_a3m_to_yaml import (
    split_colab_a3m_write_yaml,
)
from MSA.script.colab_msa_template_search.parse_fasta import parse_fasta
from MSA.local_template._common import resolve_template_engine
from MSA.db_paths import DB_PATHS_YAML, RNA_DB_ROOT
from thalkak import get_logger, run_logged, log_stream

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MSA_CONFIG_DEFAULTS = {
    "mmseqs_local": os.path.join(ROOT, "examples", "msa_config.mmseqs.yaml"),
    "hhblits_local": os.path.join(ROOT, "examples", "msa_config.hhblits.yaml"),
    "mmseqs_hhblits_local": os.path.join(ROOT, "examples", "msa_config.combined.yaml"),
}

log = get_logger("msa")


def _is_method_entity_input(path):
    try:
        with open(path) as f:
            content = yaml.safe_load(f)
    except Exception:
        return False
    return isinstance(content, dict) and ("Method" in content or "Entity" in content)


def resolve_msa_config_path(args):
    """Resolve the per-mode MSA config yaml for this invocation (local modes only).

    `--msa_config <path>` overrides; omitted → the per-mode default
    (msa_config.{mmseqs,hhblits,combined}.yaml). Config is selected
    per-invocation by this flag only (no env-var fallback), so it can't leak
    across shells/cron and stays visible in ps/slurm logs. Non-local modes
    (colab) use no local config → returns None.
    """
    if args.msa not in MSA_CONFIG_DEFAULTS:
        return None  # colab / non-local: no local config yaml
    path = args.msa_config or MSA_CONFIG_DEFAULTS[args.msa]
    if not os.path.isfile(path):
        raise FileNotFoundError(f"MSA config yaml missing: {path}")
    return os.path.abspath(path)


def _load_msa_config(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"MSA config yaml missing: {path} (required).")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _normalize_template_config(msa_cfg, msa_name):
    """Return the template-search knobs that affect reusable MSA outputs."""
    tmpl = dict(msa_cfg.get("template", {}) or {})
    return {
        "enable": bool(tmpl.get("enable", True)),
        "engine": resolve_template_engine(msa_name, tmpl),
        "query_a3m": tmpl.get("query_a3m"),
        "max_date": str(tmpl.get("max_date", "3000-01-01")),
        "max_hits": int(tmpl.get("max_hits", 20)),
    }


def _write_local_protein_inputs(protein_entities, target, dest_dir):
    """Write a protein-only, per-entity FASTA + return a contiguous protein
    stoi string in the exact form MSA.local_msa.common.input.parse_inputs
    requires (one record per unique entity; chain letters A,B,C... positional;
    stoi keys contiguous from A). `protein_entities` = parse_fasta's 3rd return
    [{"sequence","copy","orig_chain"}, ...] in original FASTA order. Original
    chain letters are intentionally DROPPED + re-labeled A,B,C... so a complex
    whose RNA chain came first still yields contiguous protein keys.
    """
    import string

    if len(protein_entities) > len(string.ascii_uppercase):
        raise ValueError(
            f"local protein input supports at most 26 unique protein entities; "
            f"got {len(protein_entities)}"
        )
    protein_fasta_path = os.path.abspath(
        os.path.join(dest_dir, f"{target}_protein.fasta")
    )
    lines, stoi_tokens = [], []
    for i, e in enumerate(protein_entities):
        lines.append(f">{target}_p{i}\n{e['sequence']}")
        stoi_tokens.append(f"{string.ascii_uppercase[i]}{int(e['copy'])}")
    with open(protein_fasta_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return protein_fasta_path, "".join(stoi_tokens)


def _resolve_custom_a3m(args, protein_entities):
    """Validate `--msa custom`'s a3m and return its absolute path.

    The file is treated exactly like one colabfold wrote, so it must carry
    ColabFold's `#<lengths>\t<copies>` header: split_colab_a3m_write_yaml reads
    the chain lengths and copy counts from it to cut the per-chain paired /
    unpaired alignments. A header that disagrees with the declared entities would
    still split cleanly, into alignments for the wrong sequences, so it is
    checked here rather than left to fail downstream."""
    src = getattr(args, "a3m_path", None)
    if not src:
        raise SystemExit(
            "--msa custom needs the a3m to use (--a3m_path, or Method.a3m_path "
            "in a full-mode input)."
        )
    src = os.path.abspath(src)
    if not os.path.isfile(src):
        raise SystemExit(f"a3m_path is not a file: {src}")

    with open(src) as f:
        header = f.readline()
    if not header.startswith("#"):
        raise SystemExit(
            f"{src}: --msa custom expects a ColabFold-format a3m, whose first "
            f"line is '#<comma-separated chain lengths>\t<comma-separated copy "
            f"counts>'. Got: {header.strip()[:60]!r}"
        )
    fields = header[1:].strip().split("\t")
    try:
        lengths = [int(x) for x in fields[0].split(",")]
        counts = (
            [int(x) for x in fields[1].split(",")] if len(fields) > 1
            else [1] * len(lengths)
        )
    except ValueError:
        raise SystemExit(
            f"{src}: could not read chain lengths / copy counts from the a3m "
            f"header line {header.strip()[:60]!r}."
        )

    want_lengths = [len(e["sequence"]) for e in protein_entities]
    want_counts = [int(e["copy"]) for e in protein_entities]
    if lengths != want_lengths or counts != want_counts:
        raise SystemExit(
            f"{src}: the a3m header describes lengths {lengths} x copies "
            f"{counts}, but the input declares lengths {want_lengths} x copies "
            f"{want_counts}. The a3m must cover the same protein entities, in "
            f"the same order."
        )
    return src


def msa_generation(args):
    from MSA.local_msa.common.input import normalize_stoi
    from MSA.local_msa.common.caps import live_caps, normalize_caps
    from MSA.local_msa.common.leveled_merge import (
        merge_key_matches,
        resolve_merge_cfg,
    )

    base_dir = (
        os.path.abspath(args.output_dir)
        if args.output_dir
        else os.path.dirname(os.path.abspath(args.seq))
    )
    # parse_fasta writes <target>_parsed.fa into base_dir below, before msa_dir
    # is created — so a fresh `--output_dir <new>` must exist first. Automated
    # callers usually mkdir -p it already (no-op here); this covers ad-hoc calls.
    os.makedirs(base_dir, exist_ok=True)
    target_name = os.path.basename(args.seq).split(".")[0]
    # The CLI passes a legacy stoi string ("A1B1"); normalize + validate it here
    # (UNK->A1, canonical chain order) for parse_fasta, the skip-key, method_log,
    # and the --stoi handed to the MSA.local_msa subprocess.
    stoi_str = normalize_stoi(args.stoi)
    cfg_path = resolve_msa_config_path(args)

    log.info("Generating MSA...")
    with log_stream(log):
        parsed_fasta_path, na_chains, protein_entities = parse_fasta(
            args.seq, stoi_str, base_dir
        )

    msa_dir = os.path.join(base_dir, "msa", args.msa)
    os.makedirs(msa_dir, exist_ok=True)

    # Prior run's MSA params (read before method_log.yaml is rewritten below) so
    # the RNA MSA search can be skipped when seq/stoi/method are unchanged — e.g.
    # running several structure models on the same target.
    _prior_path = os.path.join(msa_dir, "method_log.yaml")
    _prior = {}
    if os.path.exists(_prior_path):
        with open(_prior_path) as f:
            _prior = yaml.safe_load(f) or {}
    params_match = (
        _prior.get("msa") == args.msa
        and _prior.get("seq") == args.seq
        and _prior.get("stoi") == stoi_str
    )

    # Check if there are protein chains to search
    with open(parsed_fasta_path, "r") as f:
        protein_seq = f.read().split("\n", 1)[1].strip()
    has_protein = len(protein_seq) > 0

    msa_cfg = None  # also read after this block, when writing template_config
    if has_protein:
        msa_cfg = _load_msa_config(cfg_path) if cfg_path else None
        method_log_path = os.path.join(msa_dir, "method_log.yaml")
        existing_a3m = glob.glob(os.path.join(msa_dir, "*.a3m"))
        skip_msa = False
        config_changed = False
        recorded_dbs = config_dbs = None
        if os.path.exists(method_log_path) and existing_a3m:
            method_log = yaml.safe_load(open(method_log_path)) or {}
            base_match = (
                method_log.get("msa") == args.msa
                and method_log.get("seq") == args.seq
                and method_log.get("stoi") == stoi_str
            )
            if base_match:
                if args.msa == "colab":
                    skip_msa = True
                elif args.msa in ("mmseqs_local", "hhblits_local", "mmseqs_hhblits_local"):
                    if msa_cfg is None:
                        # These three ARE the MSA_CONFIG_DEFAULTS keys, so cfg_path
                        # — and msa_cfg — is always set here unless a mode was added
                        # to one of the two lists and not the other.
                        raise RuntimeError(
                            f"{args.msa} has no entry in MSA_CONFIG_DEFAULTS"
                        )
                    recorded_dbs = method_log.get("dbs")
                    if args.msa == "mmseqs_hhblits_local":
                        config_dbs = {
                            "mmseqs": (msa_cfg.get("mmseqs") or {}).get("dbs"),
                            "hhblits": (msa_cfg.get("hhblits") or {}).get("dbs"),
                        }
                    else:
                        config_dbs = msa_cfg.get("dbs")
                    dbs_match = recorded_dbs is not None and recorded_dbs == config_dbs
                    merge_now = resolve_merge_cfg(msa_cfg)
                    merge_match = merge_key_matches(method_log.get("merge"), merge_now)
                    if merge_now["mode"] == "leveled":
                        dedup_match = True
                    elif args.msa == "mmseqs_local":
                        recorded_dedup = (method_log.get("dedup") or {}).get(
                            "mode", "none"
                        )
                        config_dedup = (msa_cfg.get("dedup") or {}).get(
                            "mode", "raw_seq"
                        )
                        dedup_match = recorded_dedup == config_dedup
                    elif args.msa == "mmseqs_hhblits_local":
                        recorded_dedup = (method_log.get("dedup") or {}).get(
                            "mode", "none"
                        )
                        config_dedup = (
                            (msa_cfg.get("mmseqs") or {}).get("dedup") or {}
                        ).get("mode", "raw_seq")
                        dedup_match = recorded_dedup == config_dedup
                    else:
                        dedup_match = True
                    recorded_caps = method_log.get("caps")
                    caps_match = recorded_caps is not None and live_caps(
                        recorded_caps, merge_now["mode"]
                    ) == live_caps(normalize_caps(msa_cfg), merge_now["mode"])
                    template_now = _normalize_template_config(msa_cfg, args.msa)
                    template_match = method_log.get("template_config") == template_now
                    if (
                        dbs_match
                        and dedup_match
                        and caps_match
                        and template_match
                        and merge_match
                    ):
                        skip_msa = True
                    else:
                        config_changed = True

        main_a3m = os.path.join(msa_dir, f"{target_name}.a3m")
        if skip_msa:
            log.info(
                "MSA already generated with the same parameters, skipping MSA generation."
            )
            output_yaml = os.path.join(msa_dir, f"{target_name}.yaml")
            if not os.path.exists(output_yaml):
                output_yaml = split_colab_a3m_write_yaml(main_a3m)
        else:
            if config_changed:
                log.info(
                    f"MSA config changed since last run "
                    f"(dbs recorded {recorded_dbs} != config {config_dbs}, "
                    f"or dedup/caps/template/merge differ); rebuilding {msa_dir}"
                )
                shutil.rmtree(msa_dir)
                os.makedirs(msa_dir, exist_ok=True)
            for stale in [
                main_a3m,
                os.path.join(msa_dir, f"{target_name}.pickle"),
            ] + glob.glob(os.path.join(msa_dir, "*_msa_chains_*.a3m")):
                if os.path.exists(stale):
                    os.remove(stale)
            match args.msa:
                case "colab":
                    log.info("Running MSA generation using Colab...")
                    run_logged(
                        f"colabfold_batch --msa-only --templates "
                        f"{parsed_fasta_path} {msa_dir}",
                        log,
                    )
                    if not os.path.exists(main_a3m):
                        log.warning(
                            "colabfold produced no a3m (likely template/hhsearch "
                            "failure); retrying without templates."
                        )
                        for stale_t in [os.path.join(msa_dir, "pdb70.m8")] + glob.glob(
                            os.path.join(msa_dir, "templates_*")
                        ):
                            if os.path.isdir(stale_t):
                                shutil.rmtree(stale_t, ignore_errors=True)
                            elif os.path.exists(stale_t):
                                os.remove(stale_t)
                        run_logged(
                            f"colabfold_batch --msa-only "
                            f"{parsed_fasta_path} {msa_dir}",
                            log,
                        )
                    output_msa = main_a3m
                    output_yaml = split_colab_a3m_write_yaml(output_msa)
                    log.info(f"MSA generated at: {output_msa}")
                case "custom":
                    custom_a3m = _resolve_custom_a3m(args, protein_entities)
                    log.info(f"Using the supplied a3m as the protein MSA: {custom_a3m}")
                    shutil.copyfile(custom_a3m, main_a3m)
                    output_yaml = split_colab_a3m_write_yaml(main_a3m)
                case "mmseqs_local" | "hhblits_local" | "mmseqs_hhblits_local":
                    protein_fasta_path, protein_stoi = _write_local_protein_inputs(
                        protein_entities, target_name, msa_dir
                    )
                    run_logged(
                        [
                            sys.executable,
                            "-m",
                            "MSA.local_msa",
                            "--yaml",
                            cfg_path,
                            "--mode",
                            args.msa,
                            "--seq",
                            protein_fasta_path,
                            "--stoi",
                            protein_stoi,
                            "--msa_dir",
                            msa_dir,
                            "--target",
                            target_name,
                        ],
                        log,
                        cwd=ROOT,
                    )
                    output_yaml = os.path.join(msa_dir, f"{target_name}.yaml")
                case _:
                    raise ValueError(f"unhandled --msa mode: {args.msa!r}")

        with open(output_yaml, "r") as f:
            yaml_content = yaml.safe_load(f)
    else:
        if args.msa == "custom":
            raise SystemExit(
                "--msa custom supplies the protein MSA, but this target has no "
                "protein chains. Use a protein-capable msa mode, or drop a3m_path."
            )
        log.info("No protein chains found, skipping protein MSA generation.")
        yaml_content = {"a3m": []}

    method_log_path = os.path.join(msa_dir, "method_log.yaml")
    if os.path.exists(method_log_path):
        method_log = yaml.safe_load(open(method_log_path)) or {}
    else:
        method_log = {}
    method_log["msa"] = args.msa
    method_log["seq"] = args.seq
    method_log["stoi"] = stoi_str
    if has_protein and msa_cfg is not None:
        method_log["template_config"] = _normalize_template_config(msa_cfg, args.msa)
    if yaml_content.get("templates"):
        method_log["templates"] = yaml_content["templates"]
    with open(method_log_path, "w") as f:
        yaml.dump(method_log, f, sort_keys=False)

    # Add NA chains to data yaml
    yaml_content["method_log"] = os.path.join(msa_dir, "method_log.yaml")
    if na_chains:
        from MSA.script.RNA_MSA_search.sto_to_a3m import convert as sto_to_a3m

        rna_msa_script = os.path.join(ROOT, "MSA", "script", "RNA_MSA_search")
        rna_db_dir = str(RNA_DB_ROOT)
        if any(na["type"] == "rna" for na in na_chains):
            missing = [
                t
                for t in ("nhmmer", "hmmbuild", "hmmalign", "esl-sfetch")
                if shutil.which(t) is None
            ]
            if missing:
                raise RuntimeError(
                    "RNA chain(s) present but RNA MSA toolchain not on PATH: "
                    f"{', '.join(missing)}. Activate an env providing HMMER/Easel "
                    "(e.g. `conda activate thalkak`) before running RNA/RNP MSA."
                )
            missing_db = [
                str(p)
                for p in (
                    RNA_DB_ROOT / "rfam" / "rfam_v_latest.mdf",
                    RNA_DB_ROOT / "rfam" / "rfam_clust_rep_seq.fasta",
                    RNA_DB_ROOT / "rfam" / "rfam_clust_rep_seq.fasta.ssi",
                    RNA_DB_ROOT / "rnacentral" / "rnacentral_v_latest.mdf",
                    RNA_DB_ROOT / "rnacentral" / "rnacentral_clust_rep_seq.fasta",
                    RNA_DB_ROOT / "rnacentral" / "rnacentral_clust_rep_seq.fasta.ssi",
                )
                if not p.exists()
            ]
            if missing_db:
                raise RuntimeError(
                    "RNA chain(s) present but the RNA MSA databases are missing:\n  "
                    + "\n  ".join(missing_db)
                    + f"\nThey are read from `rna_root` in {DB_PATHS_YAML}. "
                    "Install them with:\n"
                    "  ./install_db.sh --family rna"
                )
        for i, na in enumerate(na_chains):
            na_fa_path = os.path.join(msa_dir, f"{target_name}_na_{i}.fa")
            with open(na_fa_path, "w") as f:
                f.write(f">{target_name}_na_{i}\n{na['sequence']}\n")

            # Run RNA MSA search and convert to a3m
            unpaired_path = os.path.abspath(na_fa_path)
            if na["type"] == "rna":
                na_sto_path = os.path.join(msa_dir, f"{target_name}_na_{i}.sto")
                na_a3m_path = os.path.join(msa_dir, f"{target_name}_na_{i}.a3m")
                # Reuse an existing RNA MSA when the same (msa, seq, stoi) already
                # produced a non-empty a3m (e.g. a prior structure-model run on
                # this target). The search is deterministic, so this only skips
                # redundant work.
                if (
                    params_match
                    and os.path.exists(na_a3m_path)
                    and os.path.getsize(na_a3m_path) > 0
                ):
                    log.info(
                        f"RNA MSA already generated for chain {i} with the same "
                        f"parameters, skipping RNA MSA search."
                    )
                else:
                    log.info(f"Running RNA MSA search for chain {i}...")
                    run_logged(
                        f"python {rna_msa_script}/msa_gen.py "
                        f"--query {na_fa_path} "
                        f"--db_dir {rna_db_dir} "
                        f"--output {na_sto_path}",
                        log,
                        cwd=msa_dir,
                    )
                    with log_stream(log):
                        sto_to_a3m(na_sto_path, na_a3m_path)
                if os.path.exists(na_a3m_path):
                    unpaired_path = os.path.abspath(na_a3m_path)
                else:
                    unpaired_path = (
                        na_fa_path  # Fallback to fasta if a3m generation fails
                    )

            yaml_content["a3m"].append(
                {
                    "paired_path": None,
                    "unpaired_path": unpaired_path,
                    "copy": na["copy"],
                    "type": na["type"],
                }
            )
    data_yaml = f"{base_dir}/{target_name}.yaml"
    # Only inputs are tagged; run_*.py takes the target name from this stem.
    if os.path.exists(data_yaml) and _is_method_entity_input(data_yaml):
        stamped = (
            f"{base_dir}/{target_name}"
            + datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
            + ".yaml"
        )
        log.warning(f"{data_yaml} is a Method+Entity input; writing to {stamped}.")
        data_yaml = stamped
    with open(data_yaml, "w") as f:
        f.write("# Fill in the following fields before running structure prediction\n")
        f.write("# job_name:\n")
        f.write("# output_dir:\n")
        f.write("# seed:\n\n")
        yaml.dump(yaml_content, f, indent=2)

    return data_yaml


if __name__ == "__main__":
    from thalkak import setup_logging

    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--msa",
        type=str,
        required=True,
        choices=["colab", "mmseqs_local", "hhblits_local", "mmseqs_hhblits_local"],
        help="MSA engine. Other knobs live in the MSA config yaml "
        "(see --msa_config; default examples/msa_config.<mode>.yaml).",
    )
    parser.add_argument(
        "--seq",
        type=str,
        required=True,
        help="Path to the input sequence file in FASTA format",
    )
    parser.add_argument(
        "--stoi",
        type=str,
        required=True,
        help="Stoichiometry information, e.g. 'A1' for one chain A",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: same as input FASTA directory)",
    )
    parser.add_argument(
        "--msa_config",
        type=str,
        default=None,
        help="Path to MSA config yaml (default: examples/msa_config.<mode>.yaml). "
        "Pass to swap the whole knob set (db list / search params / "
        "template + plot toggles) for this invocation.",
    )
    args = parser.parse_args()
    msa_generation(args)
