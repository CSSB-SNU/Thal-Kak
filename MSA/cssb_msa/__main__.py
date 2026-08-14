"""CSSB MSA pipeline subprocess entry point.

Reads the project-default yaml + runtime path fields, dispatches to
the chosen backend (mmseqs_cssb → build_a3m, hhblits_cssb →
build_a3m_hhblits, mmseqs_hhblits_cssb → run both + mmseqs-priority
merge), then runs local template search + plotting per yaml. Owns
everything cssb-specific that used to sit in msa_generation.py.

Invoked by MSA/msa_generation.py as:

    python -m MSA.cssb_msa \
        --yaml <repo>/examples/msa_config.<mode>.yaml \
        --mode {mmseqs_cssb | hhblits_cssb | mmseqs_hhblits_cssb} \
        --seq <abspath/fasta> \
        --stoi <colab-style string, e.g. "A1B1"> \
        --msa_dir <base>/msa/<mode> \
        --target <fasta stem>

The yaml is the only knobs source. No CLI flag overrides any yaml key.
"""

import argparse
import datetime
import logging
import os
import platform
import shutil
import subprocess
from pathlib import Path

import yaml

from MSA.cssb_msa.common.caps import normalize_caps, validate_caps
from MSA.cssb_msa.common.db_registry import (
    DEFAULT_REGISTRY,
    PRIMARY_UNIREF_DB,
    select,
)
from MSA.cssb_msa.common.leveled_merge import resolve_merge_cfg
from MSA.script.colab_msa_template_search.colab_a3m_to_yaml import (
    split_colab_a3m_write_yaml,
)

logger = logging.getLogger(__name__)


def _safe_cmd(cmd):
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        return out.splitlines()[0] if out else "unknown"
    except Exception:
        return "unknown"


def _log_run_metadata(cfg, mode):
    """Dump provenance to logger.info — replaces the per-script bash meta
    header so there is a single source of truth in the main log file."""
    # Report the binary the build ACTUALLY uses. Best-effort: if mmseqs is
    # absent the build fails loud at its first real mmseqs call anyway.
    try:
        from MSA.cssb_msa.mmseqs.runner import resolve_mmseqs
        mmseqs = str(resolve_mmseqs())
    except Exception:
        mmseqs = "unknown"
    py = shutil.which("python") or "unknown"
    git_dirty = _safe_cmd(["bash", "-c", "git status --porcelain 2>/dev/null | wc -l"])
    mem_avail = _safe_cmd(
        ["bash", "-c", "awk '/MemAvailable/ {printf \"%.1f\", $2/1024/1024}' /proc/meminfo"]
    )
    logger.info("=== run metadata ===")
    logger.info("mode=%s", mode)
    logger.info("host=%s user=%s cwd=%s",
                platform.node(), os.environ.get("USER", "?"), os.getcwd())
    logger.info("git_commit=%s git_branch=%s git_dirty_files=%s",
                _safe_cmd(["git", "rev-parse", "HEAD"]),
                _safe_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
                git_dirty)
    logger.info("mmseqs=%s mmseqs_version=%s",
                mmseqs, _safe_cmd([mmseqs, "version"]) if mmseqs != "unknown" else "unknown")
    logger.info("python=%s python_version=%s conda_env=%s",
                py, platform.python_version(),
                os.environ.get("CONDA_DEFAULT_ENV", "none"))
    logger.info("nproc=%s mem_avail_gib=%s",
                _safe_cmd(["nproc"]), mem_avail)
    from MSA.db_paths import LOCALDB_ROOT
    logger.info("localdb_root=%s (exists=%s)", LOCALDB_ROOT, LOCALDB_ROOT.is_dir())
    logger.info("config=%s", cfg)
    logger.info("=== end metadata ===")


def _validate_mmseqs_block(block, where):
    """Validate an mmseqs-engine config block (dbs + search + optional dedup).

    `where` is the dotted yaml location for error messages (e.g. 'mmseqs_cssb'
    or 'mmseqs_hhblits_cssb.mmseqs')."""
    if "dbs" not in block or not block["dbs"]:
        raise KeyError(f"yaml {where}.dbs must be a non-empty list")
    # Name every key against the registry before anything expensive starts, so a
    # typo in `dbs` reports itself with the list of valid keys instead of
    # surfacing as a bare KeyError from the kind lookup further down.
    select(block["dbs"])
    if block["dbs"][0] != PRIMARY_UNIREF_DB:
        raise ValueError(
            f"yaml {where}.dbs[0] must be {PRIMARY_UNIREF_DB!r}, got "
            f"{block['dbs'][0]!r}. The first entry is the Stage A profile DB and "
            f"the multimer pairing DB, and only {PRIMARY_UNIREF_DB} is supported "
            f"there. Keep it first and list the rest after it; to run UniRef100 as "
            f"the primary UniRef pass, use --msa hhblits_cssb."
        )
    if "search" not in block:
        raise KeyError(f"yaml {where}.search block is required")
    required_search = {
        "sensitivity", "filter", "diff", "db_load_mode",
        "align_eval", "qsc", "max_accept", "expand_eval",
    }
    missing = required_search - set(block["search"])
    if missing:
        raise KeyError(f"yaml {where}.search missing: {sorted(missing)}")
    # dedup is optional (defaults to raw_seq on absence): validate the value
    # only, never the presence of the block.
    dedup_mode = (block.get("dedup") or {}).get("mode", "raw_seq")
    if dedup_mode not in ("none", "raw_seq", "dedup_key"):
        raise ValueError(
            f"yaml {where}.dedup.mode must be none|raw_seq|dedup_key, "
            f"got {dedup_mode!r}"
        )


def _validate_hhblits_block(block, where, *, pair_db_read):
    """Validate an hhblits-engine config block (dbs + n_iter_* + evalue).

    `pair_db_read` says whether this block's `pair_db` reaches the builder. It
    does for `hhblits_cssb`; the combined path does not forward it (see
    `combined/build.py`), so validating it there would vouch for a key nothing
    reads."""
    if "dbs" not in block or not block["dbs"]:
        raise KeyError(f"yaml {where}.dbs must be a non-empty list")
    select(block["dbs"])  # unknown key -> named, with the valid keys listed
    for k in ("n_iter_uniref", "n_iter_env", "evalue"):
        if k not in block:
            raise KeyError(f"yaml {where}.{k} is required")
    if not pair_db_read:
        return
    # Multimer pairing runs through this DB. Check it here rather than at the
    # point of use: hhblits/build.py only touches pair_db when the target has
    # more than one unique chain, so a typo would sit unnoticed through every
    # monomer run and surface as a bare KeyError on the first heteromer.
    pair_db = block.get("pair_db", PRIMARY_UNIREF_DB)
    select([pair_db])
    if not DEFAULT_REGISTRY[pair_db].pairable:
        raise ValueError(
            f"yaml {where}.pair_db must name a pairable DB, got {pair_db!r}. "
            f"Pairing needs NCBI taxonomy plus cluster members, which only the "
            f"UniRef DBs carry; pairable keys are "
            f"{[k for k, s in DEFAULT_REGISTRY.items() if s.pairable]}."
        )


def _validate(cfg, mode):
    """Fail fast if required yaml blocks/keys are missing for the chosen mode."""
    if mode not in ("mmseqs_cssb", "hhblits_cssb", "mmseqs_hhblits_cssb"):
        raise ValueError(
            f"unknown mode {mode!r}; expected "
            f"mmseqs_cssb | hhblits_cssb | mmseqs_hhblits_cssb"
        )
    for top in ("template", "plot"):
        if top not in cfg:
            raise KeyError(f"yaml missing required top-level block: {top!r}")
    if mode == "mmseqs_cssb":
        _validate_mmseqs_block(cfg, mode)
    elif mode == "hhblits_cssb":
        _validate_hhblits_block(cfg, mode, pair_db_read=True)
    else:  # mmseqs_hhblits_cssb — nested per-source sub-blocks
        for sub in ("mmseqs", "hhblits"):
            if sub not in cfg:
                raise KeyError(f"yaml {mode}.{sub} sub-block is required")
        _validate_mmseqs_block(cfg["mmseqs"], f"{mode}.mmseqs")
        _validate_hhblits_block(cfg["hhblits"], f"{mode}.hhblits", pair_db_read=False)
    for k in ("enable", "query_a3m", "max_date", "max_hits"):
        if k not in cfg["template"]:
            raise KeyError(f"yaml template.{k} is required")
    # engine is OPTIONAL (absent -> 'auto'): validate the value only, never the
    # presence. Matches what dispatch.py's resolve_template_engine dispatches
    # on, so a bad engine fails at startup, not after the full MSA build.
    # (Combined mode ignores engine and always runs both -> absent stays 'auto'.)
    template_engine = cfg["template"].get("engine", "auto")
    if template_engine not in ("auto", "mmseqs", "hmmer"):
        raise ValueError(
            f"yaml template.engine must be auto|mmseqs|hmmer, "
            f"got {template_engine!r}"
        )
    for k in ("enable", "per_db_enable", "identity_threshold", "subsample_cap"):
        if k not in cfg["plot"]:
            raise KeyError(f"yaml plot.{k} is required")
    # Unpaired merge strategy. Absent block -> leveled with default hhfilter knobs
    # (resolve_merge_cfg owns that default and raises on a bad mode / unknown
    # hhfilter key), so a bad value fails at startup, not after the full search.
    merge_cfg = resolve_merge_cfg(cfg)
    if merge_cfg["mode"] == "leveled":
        from MSA.cssb_msa.common.hhfilter import resolve_hhfilter
        resolve_hhfilter(None)   # fail loud + early if HH-suite is not installed


def _run_combined(cfg, mode, seq, stoi, msa_dir, target):
    """Combined `mmseqs_hhblits_cssb`: run BOTH source builders + template
    engines, merge mmseqs-first, then the shared split + plotting.

    The two sources run into ``<msa_dir>/_src_{mmseqs,hhblits}/`` (each a
    self-contained single-source run); the merged master a3m + consolidated
    ``<target>_env/`` land directly under ``msa_dir`` for the shared downstream.
    """
    from MSA.cssb_msa.combined.build import (
        TEMPLATE_FINAL_N,
        build_a3m_combined,
        run_combined_template_search,
    )
    from MSA.cssb_template._common import _resolve_a3m_root

    cfg_mode = cfg
    cfg_mm, cfg_hh = cfg_mode["mmseqs"], cfg_mode["hhblits"]

    # The combined builder does not forward the hhblits sub-block's pairing
    # knobs (`combined/build.py` calls build_a3m_hhblits without pair_db_key /
    # pair_search), so both fall back to that function's defaults.
    _ignored_pair = [k for k in ("pair_db", "pair_search") if k in cfg_hh]
    if _ignored_pair:
        print(
            f"NOTE: mmseqs_hhblits_cssb ignores hhblits.{'/'.join(_ignored_pair)}; "
            f"the combined path pairs with the builder defaults "
            f"(pair_db={PRIMARY_UNIREF_DB}). Use --msa hhblits_cssb to tune pairing."
        )
    # Combined-mode template config (self-contained in combined.yaml's `template:`).
    # The combined path runs BOTH engines (mmseqs vs BioMolDB + local hmmer);
    # `query_a3m` feeds the hmmer side, `max_date`/`max_hits` apply to both. There
    # is no `engine` knob (the combined path is not single-engine).
    tmpl = dict(cfg["template"])
    plot_cfg = cfg["plot"]

    # Fail fast (before the expensive double MSA build): the hmmer template
    # stage reads the hhblits primary-UniRef a3m via `query_a3m_source`, which
    # `_resolve_a3m_root` hard-maps to `_workdir/raw/<key>/`. The hhblits builder
    # writes that dir as `raw/<hhblits.dbs[0]>/`, so query_a3m must resolve to
    # exactly hhblits.dbs[0]; otherwise the template stage would only fail AFTER
    # both pipelines complete. Skip when templates are disabled.
    if tmpl["enable"]:
        hh_primary = cfg_hh["dbs"][0]
        seed_dir_name = _resolve_a3m_root(Path("_x"), tmpl["query_a3m"]).name
        if seed_dir_name != hh_primary:
            raise ValueError(
                f"mmseqs_hhblits_cssb: template.query_a3m={tmpl['query_a3m']!r} "
                f"resolves to dir {seed_dir_name!r}, but the hhblits primary UniRef "
                f"DB (hhblits.dbs[0]) is {hh_primary!r}. The hmmer template stage "
                f"reads _src_hhblits/_workdir/raw/{hh_primary}/, so set "
                f"template.query_a3m to match (uniref100→uniref100_2026_01, "
                f"uniref30→uniref30_2302) or change hhblits.dbs[0]."
            )

    # caps validated against the UNION of both sources' effective dbs (the
    # cap-walk inside each source already validated its own subset, but the
    # combined re-cap reads caps once; assert every effective db is priced).
    effective_dbs = list(dict.fromkeys(list(cfg_mm["dbs"]) + list(cfg_hh["dbs"])))
    kinds = {db: DEFAULT_REGISTRY[db].kind for db in effective_dbs}
    merge_cfg = resolve_merge_cfg(cfg)
    caps = normalize_caps(cfg)
    validate_caps(caps, effective_dbs, kinds, merge_cfg["mode"])

    print(
        f"Running MSA generation using mmseqs_hhblits_cssb "
        f"(mmseqs dbs={cfg_mm['dbs']} + hhblits dbs={cfg_hh['dbs']}; "
        f"merge={merge_cfg['mode']})..."
    )

    out_a3m = os.path.join(msa_dir, f"{target}.a3m")
    result = build_a3m_combined(
        cfg_mode,
        seq=seq,
        stoi=stoi,
        out_a3m=out_a3m,
        msa_dir=msa_dir,
        target=target,
        caps=caps,
        merge_mode=merge_cfg["mode"],
        hhfilter_cfg=merge_cfg.get("hhfilter"),
    )

    if tmpl["enable"]:
        # Combined mode always runs BOTH template engines; an explicit
        # single-engine `template.engine` is ignored. Warn only on a real
        # override (auto == "let the mode decide" == both, i.e. honored not
        # overridden) so the shipped combined.yaml (no `engine`) stays silent.
        _forced_engine = tmpl.get("engine")
        if _forced_engine and _forced_engine != "auto":
            print(
                f"NOTE: mmseqs_hhblits_cssb ignores template.engine="
                f"{_forced_engine!r}; combined mode always runs BOTH engines "
                f"(mmseqs+hmmer, mmseqs-first merge)."
            )
        max_date = datetime.date.fromisoformat(tmpl["max_date"])
        print(
            "Running combined template search (mmseqs vs BioMolDB + local hmmer; "
            "mmseqs-priority merge)..."
        )
        run_combined_template_search(
            result,
            msa_dir=msa_dir,
            target=target,
            query_a3m_source=tmpl["query_a3m"],
            max_template_date=max_date,
            max_hits=tmpl["max_hits"],
        )

    # Split master a3m → per-chain paired/unpaired a3ms + colab-format yaml.
    # Pass top_n_templates=TEMPLATE_FINAL_N explicitly so the split's template
    # cap can't drift from the template merge's pre-selection (both = the
    # mmseqs-first SET we wrote).
    output_yaml = str(split_colab_a3m_write_yaml(out_a3m, top_n_templates=TEMPLATE_FINAL_N))
    print(f"MSA generated at: {output_yaml}")

    if plot_cfg["enable"]:
        try:
            from MSA.cssb_msa.plot.plotting import plot_cssb_msa
            print(
                f"Rendering MSA coverage + Neff plots (merged a3m; per-DB plots "
                f"unavailable for the combined mode — sources live under _src_*/)..."
            )
            plot_cssb_msa(
                msa_dir,
                overwrite=False,
                do_per_db=False,  # per-DB raw a3ms live under _src_*/_workdir, not msa_dir
                identity_threshold=plot_cfg["identity_threshold"],
                subsample_cap=plot_cfg["subsample_cap"] or None,
            )
        except Exception as exc:
            print(f"[warn] plot_cssb_msa failed: {exc}")


def _run(cfg, mode, seq, stoi, msa_dir, target):
    """Run the chosen single-source builder + template search + plotting."""
    if mode == "mmseqs_hhblits_cssb":
        return _run_combined(cfg, mode, seq, stoi, msa_dir, target)

    # Template config: each mode's config yaml carries its own complete `template:`
    # block (mmseqs uses query_a3m=uniref30, hhblits uses uniref100). The mmseqs
    # engine ignores query_a3m but DOES apply max_date, on the RELEASE date rather
    # than the deposition date; the hmmer engine uses query_a3m as the hmmbuild seed.
    tmpl = dict(cfg["template"])
    plot_cfg = cfg["plot"]
    is_hhblits = (mode == "hhblits_cssb")

    # Per-DB hit caps: validate against the config's db set before the build.
    # (Runs here, not in _validate, so the membership check sees the resolved dbs.)
    # merge mode first: it decides which caps fields are live, and validating a
    # field this merge never reads would refuse a config that would have run.
    effective_dbs = list(cfg["dbs"])
    kinds = {db: DEFAULT_REGISTRY[db].kind for db in effective_dbs}
    merge_cfg = resolve_merge_cfg(cfg)
    caps = normalize_caps(cfg)
    validate_caps(caps, effective_dbs, kinds, merge_cfg["mode"])

    print(
        f"Running MSA generation using "
        f"{'hhblits_cssb (HHblits per-DB)' if is_hhblits else 'mmseqs_cssb (mmseqs)'} "
        f"(dbs={cfg['dbs']}; merge={merge_cfg['mode']})..."
    )

    out_a3m = os.path.join(msa_dir, f"{target}.a3m")
    builder_kwargs = dict(
        fasta=seq,
        stoi=stoi,
        out_a3m=out_a3m,
        workdir=os.path.join(msa_dir, "_workdir"),
        keep_intermediate=True,
        dbs=list(cfg["dbs"]),
        merge_mode=merge_cfg["mode"],
        hhfilter_cfg=merge_cfg.get("hhfilter"),
    )

    if is_hhblits:
        from MSA.cssb_msa.hhblits.build import build_a3m_hhblits
        # Multimer pairing reuses the mmseqs_cssb pairing pipeline verbatim
        # (mmseqs search_pair on the uniref30 MMSEQS DB; expandaln → deep paired
        # MSA). Pull the SAME search params as mmseqs_cssb so the paired block is
        # method-identical to mmseqs. pair_db defaults to the primary UniRef DB
        # and is validated (registry membership + pairable) in _validate.
        pair_db_key = cfg.get("pair_db", PRIMARY_UNIREF_DB)
        pair_search = dict(cfg.get("pair_search") or {})
        for k, v in {"num_iterations": 3, "max_seqs": 10000, "initial_eval": 0.1,
                     "expand_max_seq_id": 0.95, "pair_align_eval": 0.001,
                     "sensitivity": 8.0, "db_load_mode": 2}.items():
            pair_search.setdefault(k, v)
        build_a3m_hhblits(
            **builder_kwargs,
            caps=caps,
            n_iter_uniref=cfg["n_iter_uniref"],
            n_iter_env=cfg["n_iter_env"],
            evalue=cfg["evalue"],
            pair_db_key=pair_db_key,
            pair_search=pair_search,
        )
    else:
        from MSA.cssb_msa.mmseqs.build import build_a3m, prepare_mmseqs_search
        search = prepare_mmseqs_search(cfg["search"])
        dedup_mode = (cfg.get("dedup") or {}).get("mode", "raw_seq")
        build_a3m(
            **builder_kwargs,
            caps=caps,
            search=search,
            dedup_mode=dedup_mode,
        )

    # Local template search (gated on template.enable). Engine selection +
    # dispatch live in cssb_template/dispatch.py — single source of truth shared
    # with msa_generation's skip-key via resolve_template_engine. Combined mode
    # never reaches here (_run returns _run_combined above), so dispatch only
    # ever sees the single-engine modes.
    if tmpl["enable"]:
        from MSA.cssb_template.dispatch import run_template_search_for_msa_dir
        run_template_search_for_msa_dir(msa_dir, mode=mode, template_cfg=tmpl)

    # Split master a3m → per-chain paired/unpaired a3ms + colab-format yaml.
    output_yaml = str(split_colab_a3m_write_yaml(out_a3m))
    print(f"MSA generated at: {output_yaml}")

    # Plotting — fail-silent so a plotting bug never blocks inference.
    if plot_cfg["enable"]:
        try:
            from MSA.cssb_msa.plot.plotting import plot_cssb_msa
            print(
                f"Rendering MSA coverage + Neff plots "
                f"(per-DB plots: {'on' if plot_cfg['per_db_enable'] else 'off'})..."
            )
            plot_cssb_msa(
                msa_dir,
                overwrite=False,
                do_per_db=plot_cfg["per_db_enable"],
                identity_threshold=plot_cfg["identity_threshold"],
                subsample_cap=plot_cfg["subsample_cap"] or None,
            )
        except Exception as exc:
            print(f"[warn] plot_cssb_msa failed: {exc}")


def main():
    ap = argparse.ArgumentParser(
        description="Cssb MSA pipeline subprocess entry point."
    )
    ap.add_argument("--yaml", dest="yaml_path", type=str, required=True,
                    help="Path to examples/msa_config.<mode>.yaml.")
    ap.add_argument("--mode", type=str, required=True,
                    choices=["mmseqs_cssb", "hhblits_cssb", "mmseqs_hhblits_cssb"],
                    help="MSA engine selector.")
    ap.add_argument("--seq", type=str, required=True,
                    help="Input fasta absolute path.")
    ap.add_argument("--stoi", type=str, required=True,
                    help="Stoichiometry string (CASP-style, e.g. 'A1B1').")
    ap.add_argument("--msa_dir", type=str, required=True,
                    help="Output MSA dir (<base>/msa/<mode>).")
    ap.add_argument("--target", type=str, required=True,
                    help="Fasta stem (used as the master a3m basename).")
    args = ap.parse_args()

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

    with open(args.yaml_path) as f:
        cfg = yaml.safe_load(f) or {}

    _validate(cfg, args.mode)
    _log_run_metadata(cfg, args.mode)
    _run(cfg, args.mode, args.seq, args.stoi, args.msa_dir, args.target)


if __name__ == "__main__":
    main()
