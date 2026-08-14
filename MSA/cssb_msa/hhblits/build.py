"""HHblits-engine equivalent of `mmseqs/build.build_a3m` — mirrors its phase structure (see mapping below).

Phase mapping vs `build.build_a3m`:

  Phase 1 (inputs)           — one query FASTA per unique chain under
                               `queries/`. No mmseqs `createdb` here (hhblits
                               reads FASTA directly); the pair stage makes its
                               own qdb.
  Phase 2 (UniRef Stage A)   — HHblits 3-iter against the primary UniRef DB
                               (default uniref100_2026_01; the first entry
                               in hhblits_cssb.dbs), per chain. text a3m
                               only.
  Phase 3 (env DBs Stage C)  — HHblits 3-iter against each remaining DB,
                               taking the UniRef Stage A a3m as input
                               (iterative profile refinement built INSIDE
                               hhblits from the input a3m). The default
                               env list includes uniref30_2302 as a
                               clustered UniRef pass on top of the env
                               metagenomic DBs (kind="uniref" DBs are
                               valid env entries — see env_specs below).
                               Whatever DB Phase 2 used seeds these env
                               searches.
  Phase 4 (merge)            — per-chain unpaired merge across DBs, per
                               `merge_mode`. `leveled` (default) groups the
                               per-DB a3ms by kind (uniref/env), dedups and
                               hhfilters each group and stacks uniref on
                               top; `cap_walk` head-caps each DB's records
                               by kind walking CAP_PRIORITY. Both dedup
                               cross-DB on raw_seq (literal aa-seq) and stop
                               at the per-chain total budget (total_max - N,
                               N = paired record count). Runs AFTER Phase 5
                               pair so N is known. See
                               `common/leveled_merge.py` and
                               `common/cap_merge.py`.
  Phase 5 (pair, multimer)   — REUSE the mmseqs_cssb pairing pipeline:
                               mmseqs `search_pair_uniref30` against the
                               uniref30 MMSEQS DB (expandaln → per-species
                               members → deep paired MSA). The hhblits
                               UniRef a3m cannot pair (uniref100 hhblits DB
                               strips taxonomy from headers; uniref30
                               hhblits output is cluster reps only, no
                               member expansion), so the mmseqs pair code
                               runs verbatim, via the `pair_db_key` /
                               `pair_search` args.
  Phase 6 (assemble)         — REUSE `assemble_complex_a3m_to_file`.
  Phase 7 (reorganize)       — move the per-DB a3m/hhr under `raw/`, only
                               when `keep_intermediate=True`.
  Phase 8 (method_log)       — write `method_log.yaml` beside the output a3m.

There is no per-DB pre-filter: hhfilter runs only at Phase 4, per DB-kind
group, and only under `merge.mode: leveled`.
"""

import logging
import shutil
import tempfile
from pathlib import Path

import yaml

from MSA.cssb_msa.common.assemble import assemble_complex_a3m_to_file
from MSA.cssb_msa.mmseqs.build import BuildResult
from MSA.cssb_msa.common.cap_merge import (
    cap_merge_chain,
    collect_per_db_a3m,
    count_a3m_records,
    head_a3m_records,
)
from MSA.cssb_msa.common.db_registry import (
    DEFAULT_REGISTRY,
    MERGE_ORDER,
    PRIMARY_UNIREF_DB,
)
from MSA.cssb_msa.common.hhfilter import resolve_hhfilter
from MSA.cssb_msa.common.leveled_merge import (
    DEFAULT_HHFILTER_CFG,
    leveled_merge_chain,
)
from MSA.cssb_msa.hhblits.runner import hhblits_version
from MSA.cssb_msa.common.input import ParsedInputs, parse_inputs, format_stoi
from MSA.cssb_msa.mmseqs.runner import createdb, mmseqs_version
from MSA.cssb_msa.mmseqs.search_pair import search_pair_uniref30
from MSA.cssb_msa.hhblits.search import run_hhblits_per_chain

logger = logging.getLogger(__name__)


def _write_per_chain_query_fastas(
    parsed: ParsedInputs,
    out_dir: Path,
    target: str,
) -> list[Path]:
    """Write one FASTA per unique chain — input format hhblits expects."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for cid, seq in enumerate(parsed.unique_seqs):
        p = out_dir / f"{cid}.fa"
        p.write_text(f">{target}\n{seq}\n")
        paths.append(p)
    return paths


def build_a3m_hhblits(
    fasta: Path | str,
    stoi: str,
    out_a3m: Path | str,
    *,
    workdir: Path | str | None = None,
    threads: int = 16,
    keep_intermediate: bool = False,
    dbs: list[str],
    caps: dict,
    n_iter_uniref: int = 3,
    n_iter_env: int = 3,
    evalue: float = 0.001,
    merge_mode: str = "leveled",
    hhfilter_cfg: dict | None = None,
    pair_db_key: str = PRIMARY_UNIREF_DB,
    pair_search: dict | None = None,
    mmseqs: Path | None = None,
    hhblits: Path | None = None,
) -> BuildResult:
    """Build a single ColabFold-complex-format a3m using HHblits as the
    per-DB search engine. See module docstring for phase mapping.

    Args:
        fasta, stoi, out_a3m, workdir, threads, keep_intermediate,
            mmseqs: same semantics as `build.build_a3m`.
        dbs: ordered list of DB keys (must exist in DEFAULT_REGISTRY
            and have an installed hhblits sibling). Sourced from yaml
            `hhblits_cssb.dbs` by the caller (no default in this
            function). First entry must be a UniRef DB.
        caps: normalized caps dict (`common/caps.normalize_caps` output).
            Keys `per_kind` ({"uniref","env"} head-N record caps),
            `paired_max` (paired record cap N, query-incl), `total_max`
            (per-chain total record budget; unpaired budget = total_max - N),
            `priority` (CAP_PRIORITY-ordered db_key list). Records include
            the query (row 0). Recorded verbatim into method_log under `caps`.
        n_iter_uniref, n_iter_env: hhblits `-n` for Stage A and Stage C.
            Both default to 3.
        evalue: hhblits `-e` for both stages.
        hhblits: binary override.

    Returns:
        BuildResult(a3m, method_log, workdir, parsed).
    """
    fasta = Path(fasta)
    out_a3m = Path(out_a3m)
    out_a3m.parent.mkdir(parents=True, exist_ok=True)

    parsed = parse_inputs(fasta, stoi)
    target = fasta.stem

    workdir_user_provided = workdir is not None
    if workdir is None:
        workdir = Path(tempfile.mkdtemp(
            prefix=f"build_h_{target}_", dir=out_a3m.parent
        ))
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    log_dir = workdir / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "build_a3m_hhblits: target=%s n_unique=%d cardinality=%s "
        "is_complex=%s workdir=%s",
        target, parsed.n_unique, parsed.cardinality, parsed.is_complex,
        workdir,
    )

    # Validate DBs & their hhblits siblings up front.
    db_specs = []
    for k in dbs:
        if k not in DEFAULT_REGISTRY:
            raise KeyError(f"Unknown DB key: {k}. Known: {list(DEFAULT_REGISTRY)}")
        spec = DEFAULT_REGISTRY[k]
        if not spec.has_hhblits_db():
            raise FileNotFoundError(
                f"DB {k} has no hhblits sibling at {spec.hhblits_db_stem!r} — "
                f"expected `<stem>_cs219.ffindex` to exist. Install it with "
                f"`./install_db.sh --family hhblits {k}`, or drop {k} from dbs."
            )
        db_specs.append(spec)
    if merge_mode not in ("leveled", "cap_walk"):
        raise ValueError(
            f"merge_mode must be 'leveled' or 'cap_walk', got {merge_mode!r}"
        )
    if merge_mode == "leveled":
        # fail before the (long) searches rather than at phase 4
        resolve_hhfilter(None)
    uniref_spec = db_specs[0]
    if uniref_spec.kind != "uniref":
        raise ValueError(
            f"first dbs entry must be a UniRef DB; got "
            f"{uniref_spec.key} kind={uniref_spec.kind}"
        )
    # First entry is the Phase 2 UniRef DB; everything after is used as a
    # Phase 3 env-stage DB regardless of `kind`. This lets uniref30_2302
    # (kind="uniref") be listed as a clustered-UniRef env pass alongside
    # the metagenomic env DBs.
    env_specs = list(db_specs[1:])

    try:
        # Phase 1: per-chain query FASTAs for hhblits Stage A input.
        queries_dir = workdir / "queries"
        per_chain_queries = _write_per_chain_query_fastas(
            parsed, queries_dir, target,
        )

        # Phase 2: UniRef Stage A via hhblits.
        # Output: workdir/uniref/<cid>.a3m (mirrors mmseqs builder layout).
        uniref_a3m_dir = workdir / "uniref"
        uniref_a3ms = run_hhblits_per_chain(
            input_paths=per_chain_queries,
            db_stem=uniref_spec.hhblits_db_stem,
            out_dir=uniref_a3m_dir,
            log_dir=log_dir / "phase2_uniref",
            n_iter=n_iter_uniref,
            evalue=evalue,
            threads=threads,
            hhblits=hhblits,
        )

        # Phase 3: env DBs Stage C via hhblits, iterating from Stage A a3m.
        # Output: workdir/envs/<env_key>/env/<cid>.a3m (mirrors mmseqs layout).
        envs_root = workdir / "envs"
        for env_spec in env_specs:
            env_out_dir = envs_root / env_spec.key / "env"
            run_hhblits_per_chain(
                input_paths=uniref_a3ms,
                db_stem=env_spec.hhblits_db_stem,
                out_dir=env_out_dir,
                log_dir=log_dir / f"phase3_{env_spec.key}",
                n_iter=n_iter_env,
                evalue=evalue,
                threads=threads,
                hhblits=hhblits,
            )

        # Phase 5 (before merge): multimer pairing via the SAME pipeline that
        # mmseqs_cssb uses — mmseqs `search_pair_uniref30` against the uniref30
        # MMSEQS DB (clustered + NCBI taxonomy + cluster members), because the
        # hhblits UniRef a3m cannot pair (see module docstring). Head-cap here so
        # N (paired record count) is known before the Phase 4 unpaired merge.
        paired_paths: list[Path] | None = None
        n_paired = 0  # paired record count (query incl); 0 for monomer/homomer
        if parsed.n_unique > 1:
            pair_spec = DEFAULT_REGISTRY[pair_db_key]
            if not pair_spec.has_mmseqs_db():
                raise FileNotFoundError(
                    f"pair_db {pair_db_key} has no mmseqs sibling DB "
                    f"(dbbase/basename None) — required for search_pair_uniref30"
                )
            ps = pair_search or {}
            pair_workdir = workdir / "pair"
            pair_log_dir = log_dir / "phase5_pair"
            # createdb a multi-record query (one per unique chain), exactly as
            # mmseqs/build.py Phase 1, so search_pair's qdb has n_unique entries.
            pair_query_fa = workdir / "pair_query.fas"
            pair_query_fa.write_text(
                "".join(f">{target}\n{seq}\n" for seq in parsed.unique_seqs)
            )
            pair_qdb = workdir / "pair_qdb"
            createdb(pair_query_fa, pair_qdb,
                     log_path=pair_log_dir / "00_createdb.log", mmseqs=mmseqs)
            pair_result = search_pair_uniref30(
                qdb=pair_qdb,
                uniref_dbbase=pair_spec.dbbase,
                uniref_basename=pair_spec.basename,
                workdir=pair_workdir,
                log_dir=pair_log_dir,
                threads=threads,
                mmseqs=mmseqs,
                db_load_mode=ps.get("db_load_mode", 2),
                sensitivity=ps.get("sensitivity", 8.0),
                num_iterations=ps.get("num_iterations", 3),
                max_seqs=ps.get("max_seqs", 10000),
                initial_eval=ps.get("initial_eval", 0.1),
                expand_max_seq_id=ps.get("expand_max_seq_id", 0.95),
                pair_align_eval=ps.get("pair_align_eval", 0.001),
            )
            paired_paths = [
                pair_result.paired_a3m_dir / f"{cid}.paired.a3m"
                for cid in range(parsed.n_unique)
            ]
            for p in paired_paths:
                if not p.is_file():
                    raise RuntimeError(
                        f"pair search produced no {p}; check {pair_workdir}/_logs/"
                    )
            # pairaln --pairing-dummy-mode 1 equalizes row counts across chains;
            # assert then head-cap to paired_max (positional pairing preserved).
            counts = [count_a3m_records(p.read_text()) for p in paired_paths]
            if len(set(counts)) != 1:
                raise RuntimeError(
                    f"paired a3ms have unequal record counts {counts}; "
                    f"search_pair dummy-mode 1 should have equalized them"
                )
            n_paired = min(counts[0], caps["paired_max"])
            for p in paired_paths:
                p.write_text(head_a3m_records(p.read_text(), n_paired))

        # Phase 4: per-chain cross-DB unpaired merge, per `merge_mode`. Both
        # modes stop at total_budget = total_max - N. NOTE: workdir/uniref
        # holds dbs[0] (the Phase-2 primary, uniref100 by default); every
        # other db_key (incl. uniref30) lives under workdir/envs/<key>/env.
        merged_dir = workdir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        per_db_dir_of = {uniref_spec.key: uniref_a3m_dir}
        for env_spec in env_specs:
            per_db_dir_of[env_spec.key] = envs_root / env_spec.key / "env"
        total_budget = caps["total_max"] - n_paired
        deduped_paths: list[Path] = []
        merge_info: list[dict] = []
        # Engine-specific kind rule, used by BOTH merge modes: ONLY the Phase-2
        # primary (dbs[0], uniref100 by default) is the uniref group here. uniref30
        # sits in Phase 3 as an env DB, so DEFAULT_REGISTRY[db].kind would put it in
        # the wrong group — under `leveled` it would be hhfiltered with the wrong
        # group, and under `cap_walk` it would draw the uniref head cap instead of
        # the env one. Both merge helpers take the kind rule from the caller.
        def _hh_kind(db: str) -> str:
            return "uniref" if db == uniref_spec.key else "env"

        merge_order = [d for d in MERGE_ORDER if d in dbs]
        for cid in range(parsed.n_unique):
            if merge_mode == "leveled":
                merged_text, minfo = leveled_merge_chain(
                    cid,
                    effective_dbs=dbs,
                    per_db_dir_of=per_db_dir_of,
                    order=merge_order,
                    kind_of=_hh_kind,
                    scratch=workdir / "_hhfilter" / f"c{cid}",
                    hhfilter_cfg=hhfilter_cfg,
                    total_cap=total_budget,
                )
                merge_info.append({"chain": cid, **minfo})
                logger.info(
                    "chain %d leveled: uniref(%s) %s → env %s → %d records "
                    "(cap %d, capped=%s)",
                    cid, uniref_spec.key, minfo["kinds"]["uniref"],
                    minfo["kinds"]["env"], minfo["final_depth"], total_budget,
                    minfo["capped"],
                )
            else:
                per_db_a3m = collect_per_db_a3m(
                    cid,
                    effective_dbs=dbs,
                    per_db_dir_of=per_db_dir_of,
                    priority=caps["priority"],
                )
                merged_text = cap_merge_chain(
                    per_db_a3m,
                    kinds={k: _hh_kind(k) for k, _ in per_db_a3m},
                    per_kind_cap=caps["per_kind"],
                    total_budget=total_budget,
                    dedup_mode="raw_seq",
                )
            out_p = merged_dir / f"{cid}.a3m"
            out_p.write_text(merged_text)
            deduped_paths.append(out_p)

        # Phase 6: assemble.
        assemble_complex_a3m_to_file(parsed, deduped_paths, paired_paths, out_a3m)

        # Phase 7: optional reorganization of intermediates (mirrors build.py).
        # HHblits writes both `.a3m` (alignment) and `.hhr` (report) per
        # chain; both are useful intermediates so we move both. Source
        # `_logs/` subdirs (per-step hhblits logs) are NOT moved — they
        # belong with `_workdir/_logs/` provenance.
        if keep_intermediate:
            raw_dir = workdir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)

            def _move_hhblits_outputs(src: Path, dst: Path) -> None:
                """Move per-chain a3m + hhr from src to dst. Tolerates
                empty/missing src; skips _logs subdir."""
                dst.mkdir(parents=True, exist_ok=True)
                if not src.is_dir():
                    return
                for f in src.iterdir():
                    if f.is_file() and f.suffix in (".a3m", ".hhr"):
                        f.rename(dst / f.name)

            # UniRef hhblits a3ms+hhr → raw/<uniref_key>/
            _move_hhblits_outputs(uniref_a3m_dir, raw_dir / uniref_spec.key)
            # env DBs → raw/<env_key>/
            for env_spec in env_specs:
                src = envs_root / env_spec.key / "env"
                _move_hhblits_outputs(src, raw_dir / env_spec.key)
            # paired (multimer only) → raw/pair/
            if parsed.n_unique > 1:
                pair_dst = raw_dir / "pair"
                pair_dst.mkdir(parents=True, exist_ok=True)
                pair_src = workdir / "pair" / "paired"
                for f in pair_src.glob("*.paired.a3m"):
                    f.rename(pair_dst / f.name)
                pair_src.rmdir()
            logger.info(
                "reorganized intermediates → %s/{raw,merged}/", workdir,
            )

        # Phase 8: method_log.yaml
        method_log_path = out_a3m.parent / "method_log.yaml"
        paired_dbs = [pair_db_key] if parsed.n_unique > 1 else []
        method_log = {
            "msa": "hhblits_cssb",
            "engine": "hhblits",
            "dbs": [s.key for s in db_specs],
            "paired_dbs": paired_dbs,
            "stoi": format_stoi(
                {l: c for l, c in zip(parsed.chain_letters, parsed.cardinality)}
            ),
            "is_complex": parsed.is_complex,
            "n_unique": parsed.n_unique,
            "hhblits_version": hhblits_version(hhblits),
            "mmseqs_version": mmseqs_version(mmseqs),
            "search_params": {
                "stageA": {"iterations": n_iter_uniref, "evalue": evalue},
                "stageC": {"iterations": n_iter_env, "evalue": evalue},
            },
            "merge": (
                {"mode": "leveled", "dedup": "raw_seq",
                 "hhfilter": dict(hhfilter_cfg or DEFAULT_HHFILTER_CFG),
                 "uniref_group_db": uniref_spec.key,
                 "per_chain": merge_info}
                if merge_mode == "leveled"
                else {"mode": "cap_walk", "dedup": "raw_seq"}
            ),
            "caps": caps,
            "pair_method": (
                f"mmseqs_search_pair_{pair_db_key}" if parsed.n_unique > 1 else None
            ),
        }
        method_log_path.write_text(yaml.safe_dump(method_log, sort_keys=False))
        logger.info("method_log written → %s", method_log_path)

        result = BuildResult(
            a3m=out_a3m,
            method_log=method_log_path,
            workdir=workdir,
            parsed=parsed,
        )
    finally:
        if not workdir_user_provided and not keep_intermediate:
            shutil.rmtree(workdir, ignore_errors=True)

    return result

