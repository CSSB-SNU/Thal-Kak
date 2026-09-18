import argparse, yaml, os, csv, re, glob, shutil, logging, subprocess, sys, io, contextlib
from collections import namedtuple

ROOT = os.path.dirname(os.path.abspath(__file__))

# Method choice sets — shared by the CLI flags and the --input YAML loader
# (full_args_from_input), so both interfaces validate against one list.
MSA_CHOICES = ["colab", "custom", "mmseqs_local", "hhblits_local", "mmseqs_hhblits_local"]
STRUCTURE_CHOICES = ["boltz2", "chai1", "protenix_v1", "protenix_v2", "esmfold2"]
RELAX_CHOICES = ["none", "openmm"]
# Top-5 selection metric -> the summary-CSV column it ranks by. Every predictor
# writes these same columns, and higher is better for all of them.
TOP5_METRIC_CHOICES = ["ranking_score", "ptm", "iptm", "plddt"]
_TOP5_METRIC_COLUMN = {
    "ranking_score": "ranking_score",
    "ptm": "ptm",
    "iptm": "iptm",
    "plddt": "mean_plddt",
}

# Worked example of the full-mode input yaml — pointed at from --input's help
# and from the loader's "field is required / malformed" errors.
EXAMPLE_INPUT = "examples/input.yaml"

# Full-mode input schema: the single source of truth for the Method fields of a
# `thalkak full --input` yaml. full_args_from_input resolves an input against it
# (empty required field -> error, empty optional field -> default; choices
# validated). `structure` may be a scalar or a list (run_full loops over it);
# every other Method field takes a single value.
FullField = namedtuple("FullField", ["name", "required", "default", "choices"])
FULL_METHOD_FIELDS = [
    FullField("jobname", True, None, None),
    FullField("msa", True, None, MSA_CHOICES),
    FullField("structure", True, None, STRUCTURE_CHOICES),
    FullField("relax", True, None, RELAX_CHOICES),
    FullField("top5_metric", False, "ranking_score", TOP5_METRIC_CHOICES),
    FullField("n_seed", False, 5, None),
    FullField("seed_start", False, 1, None),
    FullField("a3m_path", False, None, None),
    FullField("template_path", False, None, None),
    FullField("msa_config", False, None, None),
    FullField("model_config", False, None, None),
    FullField("relax_config", False, None, None),
    FullField("base_dir", False, None, None),
]


def _select_top5_for_job(decoy_dir, top5_dir, metric="ranking_score"):
    """Pick the top-5 structure-validated PDBs for one prediction job, ranked by
    ``metric`` -- a column of ``*_results_summary.csv`` written next to the
    decoys (one of TOP5_METRIC_CHOICES; higher is better). Copy them as
    model_[1-5].pdb into ``top5_dir`` and write a method_log.yaml summarising the
    picks. Falls back to ranking_score if the requested column is absent."""
    from Structure.script.common.structure_validation import validation

    log = get_logger("thalkak.full")
    summary_csvs = glob.glob(os.path.join(decoy_dir, "*_results_summary.csv"))
    if not summary_csvs:
        raise FileNotFoundError(f"No *_results_summary.csv found in {decoy_dir}")
    with open(summary_csvs[0]) as f:
        rows = list(csv.DictReader(f))
    column = _TOP5_METRIC_COLUMN.get(metric, "ranking_score")
    if rows and column not in rows[0]:
        log.warning(f"no {column!r} column in the summary; using ranking_score")
        column = "ranking_score"

    def _score(row):
        try:
            return float(row[column])
        except (KeyError, TypeError, ValueError):
            return float("-inf")  # empty / non-numeric ranks last

    rows.sort(key=_score, reverse=True)

    os.makedirs(top5_dir, exist_ok=True)

    picked = []
    for row in rows:
        if len(picked) >= 5:
            break
        candidates = glob.glob(os.path.join(decoy_dir, f"*{row['seed-sample']}*.pdb"))
        if not candidates:
            continue
        candidate = candidates[0]
        if not validation(candidate):
            continue
        picked.append(candidate)

    # Every predictor writes the same confidence columns into the summary read
    # above, so key those rows by seed-sample and each pick can carry its own
    # metrics into top5_dir. The monomer-only predictors leave iptm out.
    def _num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return v

    metric_cols = ["ranking_score", "mean_plddt", "ptm", "iptm"]
    summary_by_key = {
        row["seed-sample"]: {
            c: _num(row[c]) for c in metric_cols if row.get(c) not in (None, "")
        }
        for row in rows
        if row.get("seed-sample")
    }

    method_log = {"models": {}}
    metric_rows = []  # per-pick confidence metrics -> top5_dir/metrics.csv
    for i, src in enumerate(picked, 1):
        shutil.copy(src, os.path.join(top5_dir, f"model_{i}.pdb"))
        entry = {}
        ml_src = os.path.join(os.path.dirname(src), "method_log.yaml")
        if os.path.exists(ml_src):
            with open(ml_src) as f:
                entry.update(yaml.safe_load(f) or {})
        m = re.search(r"_seed_(\d+)_sample_(\d+)\.pdb$", os.path.basename(src))
        if m:
            entry["seed"] = int(m.group(1))
            entry["sample"] = int(m.group(2))
            metric_rows.append({
                "model": f"model_{i}",
                "seed": int(m.group(1)),
                "sample": int(m.group(2)),
                **summary_by_key.get(f"seed_{m.group(1)}_sample_{m.group(2)}", {}),
            })
        method_log["models"][f"model_{i}"] = entry

    with open(os.path.join(top5_dir, "method_log.yaml"), "w") as f:
        yaml.dump(method_log, f, sort_keys=False)

    # The picks' scores belong next to the picks, so relaxation and whatever
    # reads top5_dir don't have to go back to the prediction directory.
    if metric_rows:
        with open(os.path.join(top5_dir, "metrics.csv"), "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["model", "seed", "sample"] + metric_cols,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(metric_rows)

    if len(picked) < 5:
        log.warning(
            f"only {len(picked)} valid candidate(s) for "
            f"{os.path.basename(top5_dir)} (wanted 5)"
        )
    return picked


# The standard residue each common modification derives from, used only to catch
# a modification placed on the wrong residue (SEP on a Gly, say) before a run
# burns an MSA on it. Deliberately partial: an unlisted code is left alone, since
# each predictor ships its own CCD and is the authority on what it accepts.
# CCD code -> the sequence letter(s) it may replace, from the PDB chemical
# component dictionary's `mon_nstd_parent_comp_id`. Protein, DNA and RNA reuse
# the letters A/C/G, hence the repeated keys below.
#
# A position guard only: an unlisted code passes, and whether a model's CCD can
# build the component is the model's to report.
_MOD_PARENT = {
    ccd: frozenset(parents)
    for parents, ccds in (
        # --- protein ---
        ("S", ("SEP", "SAC")),                              # phospho-, N-acetyl-
        ("T", ("TPO",)),                                    # phospho-
        ("Y", ("PTR", "TYS", "NIY")),                       # phospho-, sulfo-, nitro-
        # DOPA. The CCD parent is Tyr, but the name reads as dihydroxy-Phe.
        ("YF", ("DAH",)),
        # N6 mono/di/tri-methyl-, N6-acetyl-, NZ-carboxy-, 5-hydroxy-
        ("K", ("MLZ", "MLY", "M3L", "ALY", "KCX", "LYZ")),
        # N2-methyl-, 5-methyl-, N3,N4-dimethyl-, citrulline
        ("R", ("MMO", "AGM", "2MR", "CIR")),
        ("P", ("HYP", "HY3")),                              # 4-, 3-hydroxy-
        # sulfenic/sulfinic/sulfonic acid, S-methyl-, S-nitroso-, persulfide
        ("C", ("CSO", "CSD", "OCS", "SMC", "SNC", "CSS")),
        ("H", ("MHS", "HIC", "NEP")),                       # N1-, 4-methyl-, phospho-
        ("M", ("MSE", "FME")),                              # seleno-, N-formyl-
        ("E", ("CGU",)),                                    # gamma-carboxy-
        ("D", ("PHD",)),                                    # phospho-
        ("N", ("MEN",)),                                    # N-methyl-
        ("L", ("MLE",)),                                    # N-methyl-
        ("V", ("MVA",)),                                    # N-methyl-
        ("QE", ("PCA",)),  # pyroglutamate cyclises from Gln or Glu
        # --- RNA ---
        # 1-methyl-, 2-methyl-, N6-methyl-, N6,N6-dimethyl-, 2'-O-methyl-
        ("A", ("1MA", "2MA", "6MZ", "MA6", "A2M")),
        # N1-, N2-, N2,N2-, 7-methyl-8-hydro-, N7-, 2'-O-methyl-
        ("G", ("1MG", "2MG", "M2G", "7MG", "G7M", "OMG")),
        ("C", ("5MC", "OMC")),                              # 5-, 2'-O-methyl-
        # pseudouridine, 5,6-dihydro-, 5-methyl-, 2'-O-methyl-, 3-methyl-, 4-thio-
        ("U", ("PSU", "H2U", "5MU", "OMU", "UR3", "4SU")),
        # --- DNA (parents DA/DC/DG/DT, written with the same letters) ---
        # 5-methyl-, 5-hydroxymethyl-, 5-formyl-
        ("C", ("5CM", "5HC", "5FC")),
        ("G", ("6OG", "8OG")),                              # O6-methyl-, 8-oxo-
    )
    for ccd in ccds
}


def _parse_modifications(entity, idx, input_path):
    """Validate one polymer entity's optional `modifications` and normalise them
    to ``[{"position": <1-based int>, "ccd": <str>}, ...]``, sorted by position.

    Positions are 1-based -- the convention af3, boltz and protenix all use, and
    the one a biologist writes. Each structure runner converts from here to what
    its own predictor wants (esmfold2, for one, indexes from 0).
    """
    raw = entity.get("modifications")
    if not raw:
        return []
    where = f"{input_path}: entity #{idx}"
    if not isinstance(raw, list):
        raise SystemExit(f"{where}: 'modifications' must be a list.")
    seq = "".join(str(entity["seq"]).split()).upper()
    seq_len = len(seq)
    out, seen = [], set()
    for j, m in enumerate(raw):
        if not isinstance(m, dict) or m.get("position") is None or not m.get("ccd"):
            raise SystemExit(
                f"{where}: modification #{j} needs 'position' (1-based) and "
                f"'ccd' (e.g. {{position: 3, ccd: SEP}})."
            )
        try:
            pos = int(m["position"])
        except (TypeError, ValueError):
            raise SystemExit(
                f"{where}: modification #{j} position {m['position']!r} is not an "
                f"integer."
            )
        if not 1 <= pos <= seq_len:
            raise SystemExit(
                f"{where}: modification #{j} position {pos} is outside the "
                f"sequence (1..{seq_len})."
            )
        if pos in seen:
            raise SystemExit(f"{where}: two modifications at position {pos}.")
        seen.add(pos)
        ccd = str(m["ccd"]).strip().upper()
        if not ccd.isalnum():
            raise SystemExit(
                f"{where}: modification #{j} ccd {ccd!r} is not a CCD code "
                f"(letters/digits only, e.g. SEP)."
            )
        parents = _MOD_PARENT.get(ccd)
        if parents is not None and seq[pos - 1] not in parents:
            expected = " or ".join(sorted(parents))
            raise SystemExit(
                f"{where}: modification #{j} puts {ccd} (a modified {expected}) "
                f"on {seq[pos - 1]}{pos}. Positions are 1-based -- check it, or "
                f"drop the modification if the residue is meant to change."
            )
        out.append({"position": pos, "ccd": ccd})
    return sorted(out, key=lambda d: d["position"])


def _log_chain_assignment(polymers, ligands):
    """Log which chain letters each entity receives.

    `bonds` names atoms by chain id, and chains are handed out polymer-first
    (round-robin over copies) then ligands, so printing the mapping saves the
    user from working it out by hand.
    """
    common = os.path.join(ROOT, "Structure", "script", "common")
    if common not in sys.path:
        sys.path.insert(0, common)
    from chain_utils import CIF_CHAIN_CHARS, assign_chain_indices

    copies = [int(p.get("copy", 1)) for p in polymers]
    types = [0 if p["type"] == "protein" else 1 for p in polymers]
    per_entity = assign_chain_indices(copies, types)
    parts = []
    for i, p in enumerate(polymers):
        letters = ",".join(CIF_CHAIN_CHARS[c] for c in per_entity[i])
        parts.append(f"{letters}={p['type']}#{i + 1}")
    nxt = sum(copies)
    for j, lig in enumerate(ligands):
        c = int(lig.get("copy", 1))
        letters = ",".join(CIF_CHAIN_CHARS[x] for x in range(nxt, nxt + c))
        parts.append(f"{letters}=ligand#{j + 1}({lig.get('ccd') or 'smiles'})")
        nxt += c
    if parts:
        get_logger("thalkak.full").info(
            "Chain assignment: " + "  ".join(parts)
        )


def _parse_bonds_input(cfg, input_path):
    """Validate the input's optional top-level `Bonds` and return them as-is.

    Kept in the data yaml's own shape ({chain, residue, atom} per atom) so full
    mode only renames the key on the way through, and the runners stay the single
    place that knows each predictor's addressing.
    """
    bonds = cfg.get("Bonds")
    if not bonds:
        return None
    common = os.path.join(ROOT, "Structure", "script", "common")
    if common not in sys.path:
        sys.path.insert(0, common)
    from chain_utils import parse_bonds

    try:
        parse_bonds({"bonds": bonds})
    except ValueError as exc:
        raise SystemExit(f"{input_path}: {exc}")
    return bonds


def _apply_modifications(yaml_content, protein_mods, na_mods):
    """Attach the per-entity modifications to the data yaml's a3m entries.

    msa_generation emits a3m entries as [protein entities..., NA entities...],
    each group in input order, so the two lists zip on by entry type. The MSA
    search never sees these -- they are structure-prediction input only, which
    is why they are injected here rather than threaded through msa_generation.
    """
    p_i = n_i = 0
    for entry in yaml_content.get("a3m") or []:
        if entry.get("type") == "protein":
            mods = protein_mods[p_i] if p_i < len(protein_mods) else []
            p_i += 1
        else:
            mods = na_mods[n_i] if n_i < len(na_mods) else []
            n_i += 1
        if mods:
            entry["modifications"] = mods


def full_args_from_input(input_path):
    """Load a full-mode input yaml (Method + Entity) into a run_full args set.

    Method fields resolve against FULL_METHOD_FIELDS: an empty required field is
    an error, an empty optional field falls back to its default. `structure` may
    be a scalar or a list (run_full loops over it); every other field takes a
    single value. The Entity list is split into polymer chains (protein/dna/rna)
    -- written to a FASTA + stoi so msa_generation/parse_fasta run unchanged --
    and ligands ({smiles|ccd, copy}), carried on the returned .ligand.
    """
    import string

    log = get_logger("thalkak.full")
    with open(input_path) as f:
        cfg = yaml.safe_load(f) or {}
    method = cfg.get("Method") or {}
    entities = cfg.get("Entity") or []
    if not entities:
        raise SystemExit(
            f"{input_path}: the 'Entity' section is required (see {EXAMPLE_INPUT})."
        )

    resolved = {}
    for fld in FULL_METHOD_FIELDS:
        val = method.get(fld.name)
        if val is None or val == "" or val == []:
            if fld.required:
                raise SystemExit(
                    f"{input_path}: Method.{fld.name} is required "
                    f"(see {EXAMPLE_INPUT})."
                )
            val = fld.default
        elif fld.choices:
            for v in val if isinstance(val, list) else [val]:
                if v not in fld.choices:
                    raise SystemExit(
                        f"{input_path}: Method.{fld.name} = {v!r} is not one of "
                        f"{fld.choices}."
                    )
        resolved[fld.name] = val

    # Only `structure` sweeps: one MSA feeds every structure model, and the
    # top-5 of each is relaxed one way, so those stages take a single method.
    # That also keeps msa_config / relax_config unambiguous -- each applies to
    # the one method its stage runs.
    for fld in FULL_METHOD_FIELDS:
        if fld.name != "structure" and isinstance(resolved[fld.name], list):
            raise SystemExit(
                f"{input_path}: Method.{fld.name} takes a single value, not a "
                f"list ({resolved[fld.name]}). Only Method.structure may be a "
                f"list."
            )

    # `custom` takes the protein MSA from a3m_path instead of searching for it;
    # any other mode has nothing to do with that file, so a stray a3m_path is an
    # error rather than a setting that quietly does nothing.
    if resolved["msa"] == "custom" and not resolved["a3m_path"]:
        raise SystemExit(
            f"{input_path}: Method.msa = 'custom' requires Method.a3m_path "
            f"(the ColabFold-format a3m to use)."
        )
    if resolved["msa"] != "custom" and resolved["a3m_path"]:
        raise SystemExit(
            f"{input_path}: Method.a3m_path only applies to Method.msa = "
            f"'custom', but msa is {resolved['msa']!r}."
        )
    # template_path is an optional companion to a3m_path (custom-only): it forces
    # user-supplied templates in place of a search. Optional even for custom (an
    # a3m without templates is fine), but meaningless for the searching modes.
    if resolved["msa"] != "custom" and resolved["template_path"]:
        raise SystemExit(
            f"{input_path}: Method.template_path only applies to Method.msa = "
            f"'custom', but msa is {resolved['msa']!r}."
        )

    jobname = str(resolved["jobname"])
    base_dir = os.path.abspath(
        resolved["base_dir"] or os.path.dirname(os.path.abspath(input_path))
    )
    os.makedirs(base_dir, exist_ok=True)

    # Split entities by their explicit `type` (required): polymers
    # (protein/dna/rna) -> a FASTA + stoi for msa_generation/parse_fasta;
    # ligands -> {smiles|ccd, copy}. `copy` builds the stoi (no --stoi anymore).
    polymers, ligands = [], []
    for i, e in enumerate(entities):
        etype = str(e.get("type", "")).lower()
        if not etype:
            raise SystemExit(
                f"{input_path}: entity #{i} needs an explicit 'type' "
                f"(protein/dna/rna/ligand; see {EXAMPLE_INPUT})."
            )
        if etype == "ligand":
            spec = {}
            if e.get("smiles"):
                spec["smiles"] = e["smiles"]
            elif e.get("ccd"):
                spec["ccd"] = e["ccd"]
            else:
                raise SystemExit(
                    f"{input_path}: ligand entity #{i} needs 'smiles' or 'ccd'."
                )
            spec["copy"] = int(e.get("copy", 1))
            if e.get("modifications"):
                raise SystemExit(
                    f"{input_path}: ligand entity #{i} cannot take "
                    f"'modifications' (they apply to protein/dna/rna entities)."
                )
            ligands.append(spec)
        elif etype in ("protein", "dna", "rna"):
            if not e.get("seq"):
                raise SystemExit(f"{input_path}: {etype} entity #{i} needs 'seq'.")
            polymers.append(
                {
                    **e,
                    "type": etype,
                    "modifications": _parse_modifications(e, i, input_path),
                }
            )
        else:
            raise SystemExit(
                f"{input_path}: entity #{i} has unknown type {etype!r} "
                f"(expected protein/dna/rna/ligand)."
            )

    if not polymers:
        raise SystemExit(
            f"{input_path}: at least one protein/dna/rna entity is required."
        )
    # The MSA stoi (parse_fasta / local parse_inputs) keys chains A-Z, contiguous,
    # so unique polymer entities cap at 26 (copies per entity are unbounded).
    if len(polymers) > len(string.ascii_uppercase):
        raise SystemExit(
            f"{input_path}: {len(polymers)} unique polymer entities exceed the 26 "
            f"the MSA stoi supports (chains A-Z); reduce unique sequences."
        )

    # One FASTA record per polymer entity; the copy count rides in the stoi
    # token (parse_fasta expands copies). protein-vs-NA is classified from the
    # sequence downstream by parse_fasta; `type` only routes ligand vs polymer.
    fasta_path = os.path.join(base_dir, f"{jobname}.fa")
    records, stoi_tokens = [], []
    for i, e in enumerate(polymers):
        seq = "".join(str(e["seq"]).split())
        letter = string.ascii_uppercase[i]
        records.append(f">{jobname}_{letter}\n{seq}")
        stoi_tokens.append(f"{letter}{int(e.get('copy', 1))}")
    with open(fasta_path, "w") as f:
        f.write("\n".join(records) + "\n")
    log.info(f"Wrote polymer FASTA from input entities: {fasta_path}")
    if ligands:
        log.info(f"Parsed {len(ligands)} ligand entity(ies) from input.")

    # Split to match how msa_generation orders a3m entries (proteins, then NAs);
    # run_full zips these back onto them.
    protein_mods = [p["modifications"] for p in polymers if p["type"] == "protein"]
    na_mods = [p["modifications"] for p in polymers if p["type"] != "protein"]
    n_mods = sum(len(m) for m in protein_mods) + sum(len(m) for m in na_mods)
    if n_mods:
        log.info(f"Parsed {n_mods} residue modification(s) from input entities.")

    bonds = _parse_bonds_input(cfg, input_path)
    if bonds:
        log.info(f"Parsed {len(bonds)} covalent bond(s) from input.")
        _log_chain_assignment(polymers, ligands)

    return argparse.Namespace(
        msa=resolved["msa"],
        structure=resolved["structure"],
        relax=resolved["relax"],
        top5_metric=resolved["top5_metric"],
        seq=fasta_path,
        stoi="".join(stoi_tokens),
        a3m_path=resolved["a3m_path"],
        template_path=resolved["template_path"],
        protein_modifications=protein_mods,
        na_modifications=na_mods,
        bonds=bonds,
        msa_config=resolved["msa_config"],
        model_config=resolved["model_config"],
        relax_config=resolved["relax_config"],
        base_dir=base_dir,
        n_seed=int(resolved["n_seed"]),
        seed_start=int(resolved["seed_start"]),
        ligand=ligands or None,
        jobname=jobname,
    )


def run_full(args):
    from MSA.msa_generation import msa_generation
    from Structure.structure_prediction import structure_prediction
    from Relax.relaxation import relaxation

    log = get_logger("thalkak.full")

    # Full mode is driven entirely by the --input yaml (Method + Entity, the
    # only way to pass ligands). Resolve it into a run args set here: defaults
    # filled, required checked, ligands + FASTA/stoi materialized.
    args = full_args_from_input(args.input)

    # Only `structure` sweeps; msa and relax each run one method (enforced by
    # full_args_from_input). The one MSA feeds every structure model, and each
    # model's top-5 is relaxed the one way.
    msa = args.msa
    relax = args.relax
    structures = (
        args.structure if isinstance(args.structure, list) else [args.structure]
    )
    ligand = args.ligand  # [{smiles|ccd, copy}] from the Entity list, or None

    log.info(
        f"Running Thal-Kak full: msa={msa}, structure={structures}, "
        f"relax={relax}, input FASTA: {args.seq}"
    )
    base_dir = args.base_dir
    os.makedirs(base_dir, exist_ok=True)

    # Sweep resilience: a failure for one structure model is logged (with
    # traceback) and skipped rather than aborting the whole run. `results`
    # drives the pass/fail summary, and a non-empty failure set exits non-zero.
    results = []

    def _fail(stage, exc, **where):
        log.exception(f"{stage} failed for {where}: {exc}")
        results.append({"status": f"FAIL-{stage}", "error": repr(exc), **where})

    # MSA generation, shared by every structure model below.
    section(log, f"MSA generation ({msa})")
    try:
        msa_args = namedtuple(
            "MsaArgs",
            ["msa", "seq", "stoi", "output_dir", "msa_config", "a3m_path",
             "template_path"],
        )(
            msa=msa,
            seq=args.seq,
            stoi=args.stoi,
            output_dir=base_dir,
            msa_config=args.msa_config,
            a3m_path=args.a3m_path,
            template_path=args.template_path,
        )
        data_yaml = msa_generation(msa_args)
        # Set the model-independent data-yaml fields once: ligands (from
        # --input), the structure output dir, and the seed list. job_name is
        # rewritten per structure model below.
        with open(data_yaml, "r") as f:
            yaml_content = yaml.safe_load(f)
        if ligand:
            yaml_content["ligand"] = ligand
        # Residue modifications ride on the a3m entries; the MSA above was built
        # from the unmodified sequences, which is what the search should use.
        _apply_modifications(
            yaml_content, args.protein_modifications, args.na_modifications
        )
        # Covalent bonds pass straight through in the data yaml's own shape;
        # each runner converts to its predictor's addressing.
        if args.bonds:
            yaml_content["bonds"] = args.bonds
        yaml_content["output_dir"] = os.path.join(base_dir, "structure")
        yaml_content["seed"] = list(
            range(args.seed_start, args.seed_start + args.n_seed)
        )
    except Exception as e:
        _fail("msa", e, msa=msa)
        structures = []  # nothing downstream can run without the MSA

    for structure in structures:
        section(log, f"Structure prediction ({msa} | {structure})")
        try:
            yaml_content["job_name"] = f"{msa}_{structure}"
            with open(data_yaml, "w") as f:
                yaml.dump(yaml_content, f, indent=2)
            # A combined (model-keyed) config; structure_prediction pulls
            # this structure's section, so one file covers a sweep.
            model_config = args.model_config or os.path.join(
                ROOT, "examples", "model_config.yaml"
            )
            sp_args = namedtuple(
                "SpArgs", ["model", "data_config", "model_config"]
            )(model=structure, data_config=data_yaml, model_config=model_config)
            result_dir = structure_prediction(sp_args)
            log.info(f"Structure prediction results saved at: {result_dir}")
        except Exception as e:
            _fail("structure", e, structure=structure)
            continue

        # Per-job top-5 by the model's own confidence. Job folder name
        # mirrors the structure result dir (timestamp suffix preserved).
        decoy_dir = os.path.join(result_dir, "common")
        job_name = os.path.basename(result_dir)
        top5_dir = os.path.join(base_dir, "top5", job_name)
        try:
            section(log, f"Top-5 selection ({job_name})")
            with log_stream(log):
                _select_top5_for_job(decoy_dir, top5_dir, metric=args.top5_metric)
            log.info(f"Top-5 saved at: {top5_dir}")
        except Exception as e:
            _fail("top5", e, structure=structure)
            continue

        # Relax the top-5 in place.
        section(log, f"Relaxation ({relax})")
        try:
            relax_dir = relaxation(
                top5_dir, relax, args.relax_config, ligand_specs=ligand
            )
            log.info(f"Relaxation complete: {relax_dir}")
            results.append({"status": "OK", "structure": structure})
        except Exception as e:
            _fail("relax", e, structure=structure)

    # Pass/fail summary over every structure model attempted.
    section(log, "Full-run summary")
    ok = [r for r in results if r["status"] == "OK"]
    failed = [r for r in results if r["status"] != "OK"]
    log.info(f"{len(ok)} structure model(s) completed, {len(failed)} failed.")
    for r in failed:
        where = {k: v for k, v in r.items() if k not in ("status", "error")}
        log.warning(f"{r['status']}: {where} -> {r['error']}")
    if failed:
        sys.exit(1)


def cli():
    setup_logging()
    parser = argparse.ArgumentParser(description="Thal-Kak structure prediction pipeline")
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Pipeline mode")

    # full
    p_full = subparsers.add_parser(
        "full", help="Run the full pipeline from a Method+Entity input yaml"
    )
    p_full.add_argument(
        "--input", type=str, required=True,
        help="Path to a full-mode input yaml (Method + Entity sections). "
             "Method: jobname, msa, structure (a value or a list, run in "
             "order), relax, optional top5_metric/n_seed/seed_start/a3m_path/"
             "template_path/"
             "msa_config/model_config/relax_config/base_dir. Only structure may "
             "be a list. "
             "Entity: a list of protein/dna/rna (seq, copy) and/or ligand "
             "(smiles|ccd, copy). See " + EXAMPLE_INPUT + ".",
    )

    # msa
    p_msa = subparsers.add_parser("msa", help="Run MSA generation only")
    p_msa.add_argument(
        "--msa", type=str, required=True, choices=MSA_CHOICES,
        help="MSA generation method",
    )
    p_msa.add_argument(
        "--seq", type=str, required=True, help="Path to input FASTA file"
    )
    p_msa.add_argument(
        "--stoi", type=str, required=True, help="Stoichiometry, e.g. 'A1'"
    )
    p_msa.add_argument(
        "--a3m_path", type=str, default=None,
        help="ColabFold-format a3m to use as the protein MSA "
             "(required by --msa custom; not accepted by the other modes)",
    )
    p_msa.add_argument(
        "--template_path", type=str, default=None,
        help="YAML listing user-supplied templates (path + chain_template + "
             "chain_query per entry) to force in place of a template search "
             "(optional, --msa custom only; see examples/template.yaml)",
    )
    p_msa.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: same as input FASTA directory)",
    )
    p_msa.add_argument(
        "--msa_config", type=str, default=None,
        help="MSA config yaml for local modes (default: "
             "examples/msa_config.{mmseqs,hhblits,combined}.yaml)",
    )

    # structure
    p_sp = subparsers.add_parser("structure", help="Run structure prediction only")
    p_sp.add_argument(
        "--model", type=str, required=True, choices=STRUCTURE_CHOICES,
        help="Structure prediction model",
    )
    p_sp.add_argument(
        "--data_config", type=str, required=True, help="Path to data config yaml"
    )
    p_sp.add_argument(
        "--model_config", type=str, default=None,
        help="Model config yaml, per-model or combined (default: examples/model_config.yaml)",
    )

    # relax
    p_relax = subparsers.add_parser("relax", help="Run relaxation only")
    p_relax.add_argument(
        "--decoy_dir", type=str, required=True, help="Directory containing PDB files"
    )
    p_relax.add_argument(
        "--relax", type=str, required=True, choices=RELAX_CHOICES,
        help="Relaxation method",
    )
    p_relax.add_argument(
        "--relax_config", type=str, default=None,
        help="Relax config yaml (default: examples/{relax}.yaml)",
    )

    args = parser.parse_args()

    match args.mode:
        case "full":
            run_full(args)
        case "msa":
            from MSA.msa_generation import msa_generation
            msa_generation(args)
        case "structure":
            from Structure.structure_prediction import structure_prediction
            if not args.model_config:
                args.model_config = os.path.join(ROOT, "examples", "model_config.yaml")
            structure_prediction(args)
        case "relax":
            from Relax.relaxation import relaxation
            relaxation(args.decoy_dir, args.relax, args.relax_config)


# =============================== logging ===============================
# Only the thalkak.* tree uses this setup, so third-party INFO (numexpr, jax, ...)
# stays out. Tag column = stage for INFO, level for WARNING+. External tools are
# funnelled through the same logger via run_logged (subprocess) / log_stream.

_PKG = "thalkak"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


class _Fmt(logging.Formatter):
    def format(self, record):
        record.tag = (record.levelname if record.levelno >= logging.WARNING
                      else record.name.rsplit(".", 1)[-1])
        return super().format(record)


def setup_logging(level=logging.INFO):
    """Configure the thalkak.* logger to stdout. Idempotent."""
    os.environ.setdefault("PYTHONUNBUFFERED", "1")  # child tools stream line-by-line
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass
    pkg = logging.getLogger(_PKG)
    if pkg.handlers:
        return pkg
    pkg.setLevel(level)
    pkg.propagate = False
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(_Fmt("%(asctime)s | %(tag)-9s | %(message)s", _DATEFMT))
    pkg.addHandler(h)
    return pkg


def get_logger(name):
    """Stage logger; `name` (e.g. 'msa') fills the tag column."""
    return logging.getLogger(f"{_PKG}.{name}")


def section(log, title, width=60):
    log.info(f" {title} ".center(width, "="))


def _emit(log, level, line):
    """Log one line of external-tool output: keep only the final \\r-overwrite
    (progress bars), strip trailing space, skip if blank. The line terminator
    goes first so a CRLF ending isn't mistaken for an overwrite."""
    line = line.rstrip("\r\n").rsplit("\r", 1)[-1].rstrip()
    if line:
        log.log(level, "%s", line)


def run_logged(cmd, log=None, check=True, **kw):
    """subprocess.run-like, but stream the merged stdout+stderr through `log` one
    timestamped line at a time. Returns exit code; raises on non-zero if `check`."""
    log = log or get_logger("subprocess")
    shell = isinstance(cmd, str)
    log.info("$ %s", cmd if shell else " ".join(map(str, cmd)))
    proc = subprocess.Popen(cmd, shell=shell, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, **kw)
    # Decode by hand rather than with text=True, for the newline="\n". A progress
    # callback redraws with \r and no newline, and universal-newline mode counts
    # every redraw as its own line -- protenix's urlretrieve hook fires once per
    # block, so a single cache download is a few thousand of them. Naming \n as
    # the only terminator hands each redraw burst to _emit as one line, which
    # keeps just the last state. ("" would not do: it leaves universal newlines
    # on and only stops the translation.) Popen takes no newline argument, hence
    # the wrapper.
    stream = io.TextIOWrapper(proc.stdout, errors="replace", newline="\n")
    for line in stream:
        _emit(log, logging.INFO, line)
    code = proc.wait()
    if code:
        log.error("↳ command exited with status %d", code)
        if check:
            raise subprocess.CalledProcessError(code, cmd)
    return code


def log_lines(text, log=None, level=logging.INFO):
    """Emit already-captured text line-by-line (parallel jobs, kept contiguous)."""
    log = log or get_logger("subprocess")
    for line in (text or "").splitlines():
        _emit(log, level, line)


class _LineWriter(io.TextIOBase):
    """File-like shim: forward complete lines to a logger via _emit; delegate
    fileno/isatty to the real stream so libraries probing stdout don't crash."""
    def __init__(self, log, level, stream):
        self._log, self._level, self._stream, self._buf = log, level, stream, ""

    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            _emit(self._log, self._level, line)
        return len(s)

    def flush(self):
        _emit(self._log, self._level, self._buf)
        self._buf = ""

    def fileno(self):
        return self._stream.fileno()

    def isatty(self):
        return False


@contextlib.contextmanager
def log_stream(log=None, level=logging.INFO):
    """Route sys.stdout/stderr through `log` inside the block (imported in-process
    steps: boltz/chai/esmfold/protenix). C-level fd writes still print raw."""
    log = log or get_logger("subprocess")
    out, err = _LineWriter(log, level, sys.stdout), _LineWriter(log, level, sys.stderr)
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            yield
        finally:
            out.flush()
            err.flush()

# =======================================================================


if __name__ == "__main__":
    cli()
