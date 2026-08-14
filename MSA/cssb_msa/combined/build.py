"""Combined `mmseqs_hhblits_cssb` builder + template orchestration.

Runs BOTH source pipelines into separate sub-dirs under ``msa_dir`` (each a
self-contained single-source run, ``keep_intermediate=True`` so the per-chain
intermediates + mmseqs profile persist), then performs the Stage-2 cross-source
merge:

  MSA      — unpaired depends on ``merge.mode``:
             * ``leveled`` (default): each chain's per-DB a3ms from BOTH sources are
               grouped by DB kind (uniref / env), raw_seq-deduped, and passed
               through one staged hhfilter per group, then stacked uniref-on-top
               (`common/leveled_merge.leveled_merge_texts`). Grouping across
               sources — not merging two separately-filtered sources — is the
               point; see that module's docstring.
             * ``cap_walk``: `merge.merge_unpaired_file` per chain,
               mmseqs-priority raw_seq dedup, re-cap to caps.total_max.
             Paired is `merge.merge_paired_files` in BOTH modes (mmseqs-priority,
             caps.paired_max, never filtered). Then the shared
             `common/assemble.assemble_complex_a3m_to_file`.
  Templates — run both engines (mmseqs vs BioMolDB; local hmmer for the hhblits
             side) into the sub-dirs, then `merge_templates.merge_template_envs`
             (mmseqs-priority, hhblits non-dup fill, capped at the downstream
             top_n) into ``<msa_dir>/<target>_env/``.

Sub-dir layout (under ``msa_dir``):

    _src_mmseqs/   <target>.a3m + _workdir/{prof_res, query.fas, merged/,
                                            raw/<db_key>/, raw/pair/}
    _src_hhblits/  <target>.a3m + _workdir/{merged/, raw/<db_key>/, raw/pair/}
    _merged/       per-chain combined unpaired + paired a3ms (assemble inputs)
    _workdir/      hhfilter scratch, ``leveled`` mode only

``raw/pair/`` exists for multimers only.

This module adds no search logic of its own: it calls the two source builders,
then reads their persisted per-chain outputs and the public
``run_*_for_msa_dir`` template entrypoints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

from MSA.cssb_msa.common.assemble import assemble_complex_a3m_to_file
from MSA.cssb_msa.common.db_registry import DEFAULT_REGISTRY
from MSA.cssb_msa.common.dedup import raw_seq_key
from MSA.cssb_msa.common.hhfilter import resolve_hhfilter
from MSA.cssb_msa.common.input import ParsedInputs, parse_inputs, format_stoi
from MSA.cssb_msa.common.leveled_merge import (
    DEFAULT_HHFILTER_CFG,
    leveled_merge_texts,
)
from MSA.cssb_msa.combined.merge import merge_paired_files, merge_unpaired_file
from MSA.cssb_msa.combined.merge_templates import merge_template_envs
from MSA.cssb_msa.combined.precomputed_sources import cssb_source

logger = logging.getLogger(__name__)

TEMPLATE_FINAL_N = 4


@dataclass(frozen=True)
class CombinedBuildResult:
    out_a3m: Path
    src_mmseqs_dir: Path
    src_hhblits_dir: Path
    parsed: ParsedInputs


def build_a3m_combined(
    cfg_mode: dict,
    *,
    seq: str,
    stoi: str,
    out_a3m: str,
    msa_dir: str,
    target: str,
    caps: dict,
    merge_mode: str = "leveled",
    hhfilter_cfg: dict | None = None,
) -> CombinedBuildResult:
    """Build the merged `mmseqs_hhblits_cssb` master a3m at ``out_a3m``.

    Args:
        cfg_mode: the ``mmseqs_hhblits_cssb`` config block — must contain
            ``mmseqs`` (dbs/dedup/search, same schema as ``mmseqs_cssb``) and
            ``hhblits`` (dbs/n_iter_uniref/n_iter_env/evalue) sub-blocks.
        seq, stoi: shared inputs handed to BOTH source builders (stoi = legacy string).
        out_a3m: combined master a3m output path (``<msa_dir>/<target>.a3m``).
        msa_dir, target: combined output dir + fasta stem.
        caps: normalized caps dict (`common/caps.normalize_caps`). Stage-2 re-cap
            uses ``total_max`` (unpaired budget = total_max - N) and ``paired_max``.

    Returns:
        CombinedBuildResult with the two source sub-dirs (consumed by the
        template stage) and the parsed inputs.
    """
    if merge_mode not in ("leveled", "cap_walk"):
        raise ValueError(
            f"merge_mode must be 'leveled' or 'cap_walk', got {merge_mode!r}"
        )
    if merge_mode == "leveled":
        # fail before TWO full MSA builds rather than at Stage 2
        resolve_hhfilter(None)
    cfg_mm = cfg_mode["mmseqs"]
    cfg_hh = cfg_mode["hhblits"]
    msa_dir = Path(msa_dir)
    out_a3m = Path(out_a3m)
    src_mm = msa_dir / "_src_mmseqs"
    src_hh = msa_dir / "_src_hhblits"

    # Source 1: mmseqs (own workdir; merge_mode applies to its merged/ too,
    # which is what makes _src_mmseqs a standalone-equivalent mmseqs arm).
    from MSA.cssb_msa.mmseqs.build import build_a3m, prepare_mmseqs_search

    search = prepare_mmseqs_search(cfg_mm["search"])
    dedup_mode = (cfg_mm.get("dedup") or {}).get("mode", "raw_seq")
    logger.info("combined: building mmseqs source → %s", src_mm)
    build_a3m(
        fasta=seq,
        stoi=stoi,
        out_a3m=str(src_mm / f"{target}.a3m"),
        workdir=str(src_mm / "_workdir"),
        keep_intermediate=True,
        dbs=list(cfg_mm["dbs"]),
        caps=caps,
        search=search,
        dedup_mode=dedup_mode,
        merge_mode=merge_mode,
        hhfilter_cfg=hhfilter_cfg,
    )

    # Source 2: hhblits (own workdir; merge_mode applies to its merged/ too).
    from MSA.cssb_msa.hhblits.build import build_a3m_hhblits

    logger.info("combined: building hhblits source → %s", src_hh)
    build_a3m_hhblits(
        fasta=seq,
        stoi=stoi,
        out_a3m=str(src_hh / f"{target}.a3m"),
        workdir=str(src_hh / "_workdir"),
        keep_intermediate=True,
        dbs=list(cfg_hh["dbs"]),
        caps=caps,
        n_iter_uniref=cfg_hh["n_iter_uniref"],
        n_iter_env=cfg_hh["n_iter_env"],
        evalue=cfg_hh["evalue"],
        merge_mode=merge_mode,
        hhfilter_cfg=hhfilter_cfg,
    )

    # Stage 2: cross-source per-chain merge.
    parsed = parse_inputs(Path(seq), stoi)
    n = parsed.n_unique
    merged_dir = msa_dir / "_merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    mm_merged = src_mm / "_workdir" / "merged"
    hh_merged = src_hh / "_workdir" / "merged"
    mm_pair = src_mm / "_workdir" / "raw" / "pair"
    hh_pair = src_hh / "_workdir" / "raw" / "pair"

    # Paired first (multimer only) so N (paired record count) bounds the
    # unpaired budget. Homomers/monomers (n_unique == 1) carry no paired block
    # (ColabFold convention — assemble takes paired=None).
    combined_paired: list[Path] | None = None
    n_paired = 0
    if n > 1:
        mm_paired = [mm_pair / f"{cid}.paired.a3m" for cid in range(n)]
        hh_paired = [hh_pair / f"{cid}.paired.a3m" for cid in range(n)]
        for p in mm_paired + hh_paired:
            if not p.is_file():
                raise FileNotFoundError(
                    f"combined: expected per-chain paired a3m {p} "
                    f"(source builder ran with keep_intermediate=True?)"
                )
        combined_paired = [merged_dir / f"{cid}.paired.a3m" for cid in range(n)]
        n_paired = merge_paired_files(
            mm_paired, hh_paired, combined_paired, paired_max=caps["paired_max"]
        )
        logger.info("combined: merged paired → N=%d records/chain", n_paired)

    total_budget = caps["total_max"] - n_paired
    combined_unpaired: list[Path] = []
    merge_info: list[dict] = []

    if merge_mode == "leveled":
        hh_primary = cfg_hh["dbs"][0]
        sources = [
            cssb_source(src_mm / "_workdir", src_mm / f"{target}_env",
                        lambda db: DEFAULT_REGISTRY[db].kind, "mmseqs"),
            # engine-specific: only the hhblits Phase-2 primary is the uniref group
            cssb_source(src_hh / "_workdir", src_hh / f"{target}_env",
                        lambda db: "uniref" if db == hh_primary else "env", "hhblits"),
        ]
        chain_keys = [raw_seq_key(parsed.unique_seqs[u]) for u in range(n)]
        for s in sources:
            miss = [k for k in chain_keys if k not in s.uniref and k not in s.env]
            if miss:
                raise ValueError(
                    f"combined: source {s.label!r} is missing {len(miss)} of {n} "
                    f"chain(s) by query sequence — its _workdir/raw is incomplete"
                )
        for u, key in enumerate(chain_keys):
            texts_by_kind = {
                "uniref": [s.uniref.get(key, "") for s in sources],   # source order
                "env": [s.env.get(key, "") for s in sources],         # = priority
            }
            text, minfo = leveled_merge_texts(
                texts_by_kind,
                scratch=msa_dir / "_workdir" / "_hhfilter" / f"c{u}",
                hhfilter_cfg=hhfilter_cfg,
                total_cap=total_budget,
            )
            out_cid = merged_dir / f"{u}.a3m"
            out_cid.write_text(text)
            combined_unpaired.append(out_cid)
            merge_info.append({"chain": u, **minfo})
            logger.info(
                "combined: chain %d leveled (2 sources) uniref %s → env %s → "
                "%d records (cap %d = total_max %d - N %d, capped=%s)",
                u, minfo["kinds"]["uniref"], minfo["kinds"]["env"],
                minfo["final_depth"], total_budget, caps["total_max"], n_paired,
                minfo["capped"],
            )
    else:
        for cid in range(n):
            mm_u = mm_merged / f"{cid}.a3m"
            hh_u = hh_merged / f"{cid}.a3m"
            if not mm_u.is_file() and not hh_u.is_file():
                raise FileNotFoundError(
                    f"combined: neither source produced unpaired a3m for chain {cid} "
                    f"({mm_u} / {hh_u})"
                )
            out_cid = merged_dir / f"{cid}.a3m"
            nrec = merge_unpaired_file(mm_u, hh_u, out_cid, total_budget=total_budget)
            combined_unpaired.append(out_cid)
            logger.info(
                "combined: chain %d unpaired merged → %d records (budget %d = total_max %d - N %d)",
                cid, nrec, total_budget, caps["total_max"], n_paired,
            )

    # Assemble the combined ColabFold-complex a3m (shared).
    assemble_complex_a3m_to_file(parsed, combined_unpaired, combined_paired, out_a3m)
    logger.info("combined: assembled master a3m → %s", out_a3m)

    # method_log (parent overlays seq/stoi/template_config/templates).
    method_log = {
        "msa": "mmseqs_hhblits_cssb",
        "dbs": {"mmseqs": list(cfg_mm["dbs"]), "hhblits": list(cfg_hh["dbs"])},
        "stoi": format_stoi(
            {l: c for l, c in zip(parsed.chain_letters, parsed.cardinality)}
        ),
        "is_complex": parsed.is_complex,
        "n_unique": parsed.n_unique,
        "merge": (
            {
                "mode": "leveled",
                "dedup": "raw_seq",
                "grouping": "db_kind_across_sources",
                "source_priority": ["mmseqs", "hhblits"],
                "hhfilter": dict(hhfilter_cfg or DEFAULT_HHFILTER_CFG),
                "uniref_group_db_hhblits": cfg_hh["dbs"][0],
                "paired_records": n_paired,
                "per_chain": merge_info,
            }
            if merge_mode == "leveled"
            else {
                "mode": "cross_source",
                "dedup": "raw_seq",
                "priority": "mmseqs",
                "paired_records": n_paired,
            }
        ),
        "dedup": {"mode": dedup_mode},
        "caps": caps,
    }
    (out_a3m.parent / "method_log.yaml").write_text(
        yaml.safe_dump(method_log, sort_keys=False)
    )

    return CombinedBuildResult(
        out_a3m=out_a3m,
        src_mmseqs_dir=src_mm,
        src_hhblits_dir=src_hh,
        parsed=parsed,
    )


def run_combined_template_search(
    result: CombinedBuildResult,
    *,
    msa_dir: str,
    target: str,
    query_a3m_source: str,
    max_template_date,
    max_hits: int,
    final_n: int = TEMPLATE_FINAL_N,
) -> dict:
    """Run BOTH template engines into the source sub-dirs, then merge.

    mmseqs engine searches the persisted UniRef30 profile in ``_src_mmseqs``;
    the local hmmer engine searches the ``query_a3m_source`` per-chain a3m in
    ``_src_hhblits``. Each writes its own ``<target>_env/``; we merge them
    mmseqs-first into ``<msa_dir>/<target>_env/`` (consumed by the data-yaml
    writer's disk scan).

    Returns the per-qid merge stats from `merge_template_envs`.
    """
    from MSA.cssb_template.engine_mmseqs import run_mmseqs_for_msa_dir
    from MSA.cssb_template.engine_hmmer import run_hmmer_for_msa_dir

    logger.info("combined: mmseqs template search (vs BioMolDB) in %s", result.src_mmseqs_dir)
    run_mmseqs_for_msa_dir(
        msa_dir=result.src_mmseqs_dir,
        max_template_date=max_template_date,
        max_hits=max_hits,
    )
    logger.info(
        "combined: hmmer template search (query a3m source=%s) in %s",
        query_a3m_source, result.src_hhblits_dir,
    )
    run_hmmer_for_msa_dir(
        msa_dir=result.src_hhblits_dir,
        query_a3m_source=query_a3m_source,
        max_template_date=max_template_date,
        max_hits=max_hits,
    )

    out_env = Path(msa_dir) / f"{target}_env"
    stats = merge_template_envs(
        result.src_mmseqs_dir / f"{target}_env",
        result.src_hhblits_dir / f"{target}_env",
        out_env,
        final_n=final_n,
    )
    total = sum(s["total"] for s in stats.values())
    logger.info(
        "combined: merged templates → %s (%d rows over %d chains, mmseqs-first)",
        out_env / "pdb70.m8", total, len(stats),
    )
    return stats
