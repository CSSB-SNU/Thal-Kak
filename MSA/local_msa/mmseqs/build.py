"""`build_a3m` — the mmseqs MSA builder (public entry point).

Wires the phases together: parse inputs → `createdb` → UniRef Stage A →
env DBs Stage C (one per env DB in the config) → (multimer only) UniRef
pair search → per-chain unpaired merge → assemble ColabFold-complex a3m.
One function, one a3m output.

The caller passes the resulting a3m to
`MSA/script/colab_msa_template_search/colab_a3m_to_yaml.split_colab_a3m_write_yaml`
to get per-chain paired/unpaired splits + the data yaml — exactly the
same downstream as the colab path.
"""

import logging
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from MSA.local_msa.common.assemble import assemble_complex_a3m_to_file
from MSA.local_msa.common.db_registry import (
    DEFAULT_REGISTRY,
    MERGE_ORDER,
    PRIMARY_UNIREF_DB,
)
from MSA.local_msa.common.cap_merge import (
    cap_merge_chain,
    collect_per_db_a3m,
    count_a3m_records,
    head_a3m_records,
)
from MSA.local_msa.common.hhfilter import resolve_hhfilter
from MSA.local_msa.common.leveled_merge import (
    DEFAULT_HHFILTER_CFG,
    leveled_merge_chain,
)
from MSA.local_msa.common.input import ParsedInputs, parse_inputs, format_stoi
from MSA.local_msa.mmseqs.runner import (
    createdb,
    mmseqs_version,
    rmdb,
)
from MSA.local_msa.mmseqs.search_envdb import search_envdb_with_profile
from MSA.local_msa.mmseqs.search_pair import search_pair_uniref30
from MSA.local_msa.mmseqs.search_uniref import search_uniref30_monomer

logger = logging.getLogger(__name__)


# Defaults for the 7 optional mmseqs search knobs. A yaml that omits them still
# gets the effective values recorded into method_log.search_params.
_SEARCH_V2_DEFAULTS = {
    "num_iterations":    3,
    "max_seqs":          10000,
    "initial_eval":      0.1,
    "expand_max_seq_id": 0.95,
    "alt_ali":           10,
    "filter_min_enable": 1000,
    "pair_align_eval":   0.001,
}


def prepare_mmseqs_search(search_cfg: dict) -> dict:
    """Return a copy of the yaml ``search`` block ready for ``build_a3m``.

    Coerces ``expand_eval`` to float (yaml allows ``.inf``/``inf`` for infinity)
    and fills the 7 optional tuning defaults (``setdefault``, so present keys win).
    Single source of truth shared by the mmseqs_local single-mode path
    (``__main__._run``) and the combined path (``combined/build.py``).
    """
    search = dict(search_cfg)
    search["expand_eval"] = float(search["expand_eval"])
    for k, v in _SEARCH_V2_DEFAULTS.items():
        search.setdefault(k, v)
    return search


@dataclass(frozen=True)
class BuildResult:
    a3m: Path
    method_log: Path
    workdir: Path
    parsed: ParsedInputs


def build_a3m(
    fasta: Path | str,
    stoi: str,
    out_a3m: Path | str,
    *,
    dbs: list[str],
    search: dict,
    caps: dict,
    dedup_mode: str = "raw_seq",
    merge_mode: str = "leveled",
    hhfilter_cfg: dict | None = None,
    workdir: Path | str | None = None,
    threads: int = 16,
    keep_intermediate: bool = False,
    mmseqs: Path | None = None,
) -> BuildResult:
    """Build a single ColabFold-complex-format a3m for one query.

    Args:
        fasta: input FASTA (one record per unique entity).
        stoi: legacy stoi string, e.g. ``"A1"`` (monomer), ``"A2"`` (homomer),
            ``"A1B1"`` (heteromer); ``"UNK"`` → ``A1``.
        out_a3m: where to write the final a3m.
        dbs: ordered DB key list (each must exist in DEFAULT_REGISTRY
            and have an mmseqs sibling). First entry must be UniRef30
            (``uniref30_2302``); the function uses it for the Phase 2
            UniRef Stage A profile that env DBs chain from, and for the
            Phase 5 multimer pair search. UniRef100 (Phase 2b) runs iff
            ``uniref100_2026_01`` appears in the list. Env DBs run in
            list order. Sourced from yaml ``mmseqs_local.dbs`` by the
            caller; this function does NOT have a default.
        search: ColabFold-canonical mmseqs search params. Required keys
            ``sensitivity, filter, diff, db_load_mode, align_eval, qsc,
            max_accept, expand_eval`` (sourced from yaml
            ``mmseqs_local.search``). ``filter=True`` silently overrides
            ``align_eval/qsc/max_accept`` to 10/0.8/100000 inside the
            search modules, matching upstream ColabFold; set
            ``filter: false`` to make those three knobs effective.
        caps: normalized caps dict (``common/caps.normalize_caps`` output).
            Consumed keys ``per_kind`` ({"uniref","env"} head-N record caps),
            ``paired_max`` (paired record cap N, query-incl), ``total_max``
            (per-chain total record budget; unpaired budget = total_max - N),
            ``priority`` (CAP_PRIORITY-ordered db_key list). All counts are
            a3m records INCLUDING the query (row 0). Recorded verbatim into
            ``method_log.yaml`` under ``caps``.
        workdir: scratch dir. Default: a temp dir under
            ``out_a3m.parent``, removed on success unless
            ``keep_intermediate=True``. If you pass an explicit path
            it's used as-is and never auto-removed (assumption: you
            want to inspect).
        threads: per-stage mmseqs ``--threads``.
        keep_intermediate: keep the default temp workdir after success.
            No effect if ``workdir`` was given explicitly.
        mmseqs: binary override.
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
        workdir = Path(tempfile.mkdtemp(prefix=f"build_{target}_", dir=out_a3m.parent))
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    log_dir = workdir / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "build_a3m: target=%s n_unique=%d cardinality=%s is_complex=%s workdir=%s",
        target, parsed.n_unique, parsed.cardinality, parsed.is_complex, workdir,
    )

    # Validate dbs + look up specs.
    for k in dbs:
        if k not in DEFAULT_REGISTRY:
            raise KeyError(f"Unknown DB key: {k}. Known: {list(DEFAULT_REGISTRY)}")
        if not DEFAULT_REGISTRY[k].has_mmseqs_db():
            raise FileNotFoundError(
                f"DB {k} has no mmseqs sibling at "
                f"{DEFAULT_REGISTRY[k].dbbase!r}/{DEFAULT_REGISTRY[k].basename!r} — "
                f"drop {k} from dbs or add the mmseqs-format DB."
            )
    if merge_mode not in ("leveled", "cap_walk"):
        raise ValueError(
            f"merge_mode must be 'leveled' or 'cap_walk', got {merge_mode!r}"
        )
    if merge_mode == "leveled":
        # Resolve hhfilter NOW, not after an hour of searching: `leveled` needs the
        # HH-suite binary (from `hhsuite` in environment.yml) and the failure is
        # otherwise deferred to phase 4.
        resolve_hhfilter(None)
        if dedup_mode != "raw_seq":
            logger.warning(
                "dedup.mode=%r is INERT under merge.mode='leveled' — the leveled "
                "merge always dedups on raw_seq. The value is still recorded in "
                "method_log and is still part of the re-run skip key, so changing "
                "it forces a rebuild that cannot change the a3m. Use "
                "merge.mode='cap_walk' if you need this knob.",
                dedup_mode,
            )
    # `__main__` refuses this earlier for yaml-driven runs; this covers direct
    # build_a3m() callers.
    uniref_key = dbs[0]
    if uniref_key != PRIMARY_UNIREF_DB:
        raise ValueError(
            f"dbs[0] must be {PRIMARY_UNIREF_DB!r} (Stage A profile + pairing DB "
            f"+ the raw output dir name); got {uniref_key!r}"
        )
    include_uniref100 = "uniref100_2026_01" in dbs
    # env DBs in canonical MERGE_ORDER priority, restricted to whatever
    # is listed in `dbs`. Orders the Phase 3 env SEARCH loop. The merge order
    # is decided separately: `leveled` walks MERGE_ORDER ∩ dbs, `cap_walk`
    # walks caps.priority (CAP_PRIORITY).
    env_keys = [
        k for k in MERGE_ORDER
        if k in dbs and DEFAULT_REGISTRY[k].kind == "env"
    ]

    try:
        # Phase 1: query.fas (one record per UNIQUE seq, jobname header)
        # + createdb
        query_fa = workdir / "query.fas"
        query_fa.write_text(
            "".join(f">{target}\n{seq}\n" for seq in parsed.unique_seqs)
        )
        qdb = workdir / "qdb"
        createdb(query_fa, qdb, log_path=log_dir / "00_createdb.log", mmseqs=mmseqs)

        # Phase 2: UniRef Stage A → uniref/N.a3m + prof_res
        uniref_spec = DEFAULT_REGISTRY[uniref_key]
        t_phase = time.monotonic()
        logger.info("phase 2: UniRef Stage A search start (db=%s)", uniref_spec.basename)
        uniref_result = search_uniref30_monomer(
            qdb=qdb,
            uniref_dbbase=uniref_spec.dbbase,
            uniref_basename=uniref_spec.basename,
            workdir=workdir,
            log_dir=log_dir / "phase2_uniref",
            threads=threads,
            mmseqs=mmseqs,
            db_load_mode=search["db_load_mode"],
            sensitivity=search["sensitivity"],
            expand_eval=float(search["expand_eval"]),
            align_eval=search["align_eval"],
            diff=search["diff"],
            qsc=search["qsc"],
            max_accept=search["max_accept"],
            filter=search["filter"],
            num_iterations=search.get("num_iterations", 3),
            max_seqs=search.get("max_seqs", 10000),
            initial_eval=search.get("initial_eval", 0.1),
            expand_max_seq_id=search.get("expand_max_seq_id", 0.95),
            alt_ali=search.get("alt_ali", 10),
            filter_min_enable=search.get("filter_min_enable", 1000),
        )
        logger.info("phase 2: UniRef Stage A done in %.1fs", time.monotonic() - t_phase)
        prof_res = uniref_result.prof_res

        # Phase 2b: UniRef100 Stage A → uniref100/uniref/N.a3m + result.a3m_db.
        # Folded into the inference master a3m by the Phase 4 merge. Reuses the
        # generic Stage A function (`search_uniref30_monomer` is DB-agnostic
        # despite the name). Runs only when `uniref100_2026_01` is in `dbs`.
        uniref100_result = None
        if include_uniref100:
            uniref100_spec = DEFAULT_REGISTRY["uniref100_2026_01"]
            t_phase = time.monotonic()
            logger.info("phase 2b: UniRef100 Stage A search start (db=%s)", uniref100_spec.basename)
            uniref100_result = search_uniref30_monomer(
                qdb=qdb,
                uniref_dbbase=uniref100_spec.dbbase,
                uniref_basename=uniref100_spec.basename,
                workdir=workdir / "uniref100",
                log_dir=log_dir / "phase2b_uniref100",
                threads=threads,
                mmseqs=mmseqs,
                db_load_mode=search["db_load_mode"],
                sensitivity=search["sensitivity"],
                expand_eval=float(search["expand_eval"]),
                align_eval=search["align_eval"],
                diff=search["diff"],
                qsc=search["qsc"],
                max_accept=search["max_accept"],
                filter=search["filter"],
                num_iterations=search.get("num_iterations", 3),
                max_seqs=search.get("max_seqs", 10000),
                initial_eval=search.get("initial_eval", 0.1),
                expand_max_seq_id=search.get("expand_max_seq_id", 0.95),
                alt_ali=search.get("alt_ali", 10),
                filter_min_enable=search.get("filter_min_enable", 1000),
            )
            logger.info("phase 2b: UniRef100 Stage A done in %.1fs", time.monotonic() - t_phase)

        # Phase 3: env DBs Stage C (sequential, reusing UniRef30's prof_res).
        # env_keys was resolved from `dbs` ∩ MERGE_ORDER (env-kind only) at
        # the top of the function. env DBs follow colabfold's profile chain
        # from the primary UniRef DB only.
        envs_root = workdir / "envs"
        for env_key in env_keys:
            spec = DEFAULT_REGISTRY[env_key]
            t_phase = time.monotonic()
            logger.info("phase 3: env search start (db=%s)", env_key)
            search_envdb_with_profile(
                qdb=qdb,
                prof_res=prof_res,
                env_dbbase=spec.dbbase,
                env_basename=spec.basename,
                workdir=envs_root / env_key,
                log_dir=log_dir / f"phase3_{env_key}",
                threads=threads,
                mmseqs=mmseqs,
                expandable=spec.expandable,
                db_load_mode=search["db_load_mode"],
                sensitivity=search["sensitivity"],
                expand_eval=float(search["expand_eval"]),
                align_eval=search["align_eval"],
                diff=search["diff"],
                qsc=search["qsc"],
                max_accept=search["max_accept"],
                filter=search["filter"],
                num_iterations=search.get("num_iterations", 3),
                max_seqs=search.get("max_seqs", 10000),
                initial_eval=search.get("initial_eval", 0.1),
                expand_max_seq_id=search.get("expand_max_seq_id", 0.95),
                alt_ali=search.get("alt_ali", 10),
                filter_min_enable=search.get("filter_min_enable", 1000),
            )
            logger.info("phase 3: env search done (db=%s) in %.1fs", env_key, time.monotonic() - t_phase)

        # Phase 5: UniRef pair search — multimer only (n_unique > 1). Runs
        # BEFORE the Phase 4 merge so the paired record count N is known:
        # the unpaired budget is total_max - N. Homomers (n_unique=1,
        # cardinality>1) skip — colabfold gates paired_msa on
        # `len(query_seqs_cardinality) > 1`.
        paired_paths: list[Path] | None = None
        paired_n = [0] * parsed.n_unique  # per-cid paired record count (query incl); 0 for monomer/homomer
        if parsed.n_unique > 1:
            pair_workdir = workdir / "pair"
            t_phase = time.monotonic()
            logger.info("phase 5: UniRef pair search start (n_unique=%d)", parsed.n_unique)
            pair_result = search_pair_uniref30(
                qdb=qdb,
                uniref_dbbase=uniref_spec.dbbase,
                uniref_basename=uniref_spec.basename,
                workdir=pair_workdir,
                log_dir=log_dir / "phase5_pair",
                threads=threads,
                mmseqs=mmseqs,
                db_load_mode=search["db_load_mode"],
                sensitivity=search["sensitivity"],
                num_iterations=search.get("num_iterations", 3),
                max_seqs=search.get("max_seqs", 10000),
                initial_eval=search.get("initial_eval", 0.1),
                expand_max_seq_id=search.get("expand_max_seq_id", 0.95),
                pair_align_eval=search.get("pair_align_eval", 0.001),
            )
            logger.info("phase 5: UniRef pair search done in %.1fs", time.monotonic() - t_phase)
            paired_paths = [
                pair_result.paired_a3m_dir / f"{cid}.paired.a3m"
                for cid in range(parsed.n_unique)
            ]
            for p in paired_paths:
                if not p.is_file():
                    raise RuntimeError(
                        f"pair search produced no {p}; check {pair_workdir}/_logs/"
                    )
            # Head-cap paired a3ms to the SAME first N records across ALL
            # chains (positional pairing: row k of chain 0 pairs with row k of
            # chain 1, ...). pairaln --pairing-dummy-mode 1 equalizes row
            # counts across chains, so assert equality then clamp to
            # paired_max. Records include the query (row 0).
            paired_counts = [count_a3m_records(p.read_text()) for p in paired_paths]
            if len(set(paired_counts)) != 1:
                raise RuntimeError(
                    f"paired a3m record counts differ across chains: "
                    f"{dict(enumerate(paired_counts))}; pairaln dummy-mode 1 "
                    f"should have equalized them — check {pair_workdir}/_logs/"
                )
            n_cid = min(paired_counts[0], caps["paired_max"])
            for p in paired_paths:
                p.write_text(head_a3m_records(p.read_text(), n_cid))
            paired_n = [n_cid] * parsed.n_unique
            logger.info(
                "phase 5: paired head-cap → N=%d records/chain (paired_max=%d, raw=%d)",
                n_cid, caps["paired_max"], paired_counts[0],
            )

        # Phase 4: per-chain cross-DB unpaired merge, per `merge_mode`. Both
        # modes stop at total_budget = total_max - N (N = this chain's paired
        # record count) and write merged/<cid>.a3m, where assemble expects it.
        merged_dir = workdir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        per_db_dir_of: dict[str, Path] = {uniref_key: workdir / "uniref"}
        if include_uniref100:
            per_db_dir_of["uniref100_2026_01"] = workdir / "uniref100" / "uniref"
        for env_key in env_keys:
            per_db_dir_of[env_key] = envs_root / env_key / "env"
        kinds = {k: DEFAULT_REGISTRY[k].kind for k in per_db_dir_of}

        deduped_paths: list[Path] = []
        merge_info: list[dict] = []
        merge_order = [d for d in MERGE_ORDER if d in dbs]
        t_phase = time.monotonic()
        for cid in range(parsed.n_unique):
            total_budget = caps["total_max"] - paired_n[cid]
            if merge_mode == "leveled":
                # kind from the registry: uniref30/uniref100 are the uniref group,
                # every env DB the env group. (The hhblits builder must NOT do this
                # — only its dbs[0] is uniref there. See leveled_merge's docstring.)
                merged_text, minfo = leveled_merge_chain(
                    cid,
                    effective_dbs=dbs,
                    per_db_dir_of=per_db_dir_of,
                    order=merge_order,
                    kind_of=lambda d: DEFAULT_REGISTRY[d].kind,
                    scratch=workdir / "_hhfilter" / f"c{cid}",
                    hhfilter_cfg=hhfilter_cfg,
                    total_cap=total_budget,
                )
                merge_info.append({"chain": cid, **minfo})
                logger.info(
                    "chain %d leveled: uniref %s → env %s → %d records "
                    "(cap %d = total_max %d - N %d, capped=%s)",
                    cid, minfo["kinds"]["uniref"], minfo["kinds"]["env"],
                    minfo["final_depth"], total_budget, caps["total_max"],
                    paired_n[cid], minfo["capped"],
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
                    kinds=kinds,
                    per_kind_cap=caps["per_kind"],
                    total_budget=total_budget,
                    dedup_mode=dedup_mode,
                )
                logger.info(
                    "chain %d cap-walk: %d per-db a3m → %d records "
                    "(budget %d = total_max %d - N %d)",
                    cid, len(per_db_a3m), count_a3m_records(merged_text),
                    total_budget, caps["total_max"], paired_n[cid],
                )
            out_cid = merged_dir / f"{cid}.a3m"
            out_cid.write_text(merged_text)
            deduped_paths.append(out_cid)
        logger.info("phase 4: %s merge done in %.1fs",
                    merge_mode, time.monotonic() - t_phase)

        # Cleanup per-DB result DBs (the *.a3m mmseqs result DBs). The
        # unpacked text a3ms (workdir/uniref/, workdir/uniref100/uniref/,
        # workdir/envs/<key>/env/) remain — the Phase 4 merge consumed them
        # above and the keep_intermediate reorganize below may still move them.
        merge_log_dir = log_dir / "phase4_merge"
        merge_log_dir.mkdir(parents=True, exist_ok=True)
        rmdb(uniref_result.a3m_db,
             log_path=merge_log_dir / "04_rmdb_uniref.log", mmseqs=mmseqs)
        if uniref100_result is not None:
            rmdb(uniref100_result.a3m_db,
                 log_path=merge_log_dir / "04b_rmdb_uniref100.log", mmseqs=mmseqs)
        for env_key in env_keys:
            rmdb(envs_root / env_key / "env.a3m",
                 log_path=merge_log_dir / f"05_rmdb_{env_key}.log",
                 mmseqs=mmseqs)

        # Phase 6: assemble ColabFold complex a3m → out_a3m
        assemble_complex_a3m_to_file(parsed, deduped_paths, paired_paths, out_a3m)

        # Phase 7: reorganize per-DB raw + per-chain merged a3m into a
        # user-facing layout under workdir, gated on keep_intermediate so
        # the work is wasted only when the workdir is about to be nuked.
        # Result: workdir/raw/<db_key>/<cid>.a3m (selected inference DBs
        # + optional pair) and workdir/merged/<cid>.a3m
        # (post-dedup per-chain). The mmseqs intermediate DBs (qdb,
        # prof_res, res*, tmp, _logs) stay where they are.
        if keep_intermediate:
            raw_dir = workdir / "raw"
            # UniRef30 → raw/uniref30_2302/
            uniref30_dst = raw_dir / "uniref30_2302"
            uniref30_dst.mkdir(parents=True, exist_ok=True)
            for f in (workdir / "uniref").glob("*.a3m"):
                f.rename(uniref30_dst / f.name)
            (workdir / "uniref").rmdir()
            # UniRef100 → raw/uniref100_2026_01/. Only present when
            # uniref100_2026_01 was included in `dbs`.
            if include_uniref100:
                uniref100_dst = raw_dir / "uniref100_2026_01"
                uniref100_dst.mkdir(parents=True, exist_ok=True)
                for f in (workdir / "uniref100" / "uniref").glob("*.a3m"):
                    f.rename(uniref100_dst / f.name)
                (workdir / "uniref100" / "uniref").rmdir()
            # env DBs → raw/<env_key>/
            for env_key in env_keys:
                dst = raw_dir / env_key
                dst.mkdir(parents=True, exist_ok=True)
                for f in (workdir / "envs" / env_key / "env").glob("*.a3m"):
                    f.rename(dst / f.name)
                (workdir / "envs" / env_key / "env").rmdir()
            # paired (multimer only) → raw/pair/
            if parsed.n_unique > 1:
                pair_dst = raw_dir / "pair"
                pair_dst.mkdir(parents=True, exist_ok=True)
                for f in (workdir / "pair" / "paired").glob("*.paired.a3m"):
                    f.rename(pair_dst / f.name)
                (workdir / "pair" / "paired").rmdir()
            # workdir/merged/ is already where Phase 4 put the per-chain
            # merged a3ms (no rename needed).
            logger.info(
                "reorganized intermediates → %s/{raw,merged}/", workdir,
            )

        # Phase 8: method_log.yaml next to out_a3m
        method_log_path = out_a3m.parent / "method_log.yaml"
        paired_dbs = [uniref_key] if parsed.n_unique > 1 else []
        method_log = {
            "msa": "mmseqs_local",
            "dbs": list(dbs),
            "paired_dbs": paired_dbs,
            "stoi": format_stoi(
                {l: c for l, c in zip(parsed.chain_letters, parsed.cardinality)}
            ),
            "is_complex": parsed.is_complex,
            "n_unique": parsed.n_unique,
            "mmseqs_version": mmseqs_version(mmseqs),
            "search_params": {k: (float(v) if k == "expand_eval" else v)
                              for k, v in search.items()},
            "merge": (
                {"mode": "leveled",
                 "hhfilter": dict(hhfilter_cfg or DEFAULT_HHFILTER_CFG),
                 "per_chain": merge_info}
                if merge_mode == "leveled" else {"mode": "cap_walk"}
            ),
            "dedup": {"mode": dedup_mode},
            "caps": caps,
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

