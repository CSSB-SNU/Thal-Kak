"""Per-chain hhblits search primitive.

Exposes one function `run_hhblits_per_chain` that runs hhblits once per
chain index against a single DB and writes per-chain a3m files under a
caller-chosen out_dir using the convention `<cid>.a3m` (matching
`MSA.local_msa.mmseqs.search_uniref.search_uniref30_monomer`'s
`workdir/uniref/<cid>.a3m` layout).

Whether the input is a query FASTA (UniRef Stage A) or an a3m from a
previous stage (env DB Stage C, iterative refinement) is up to the
caller — hhblits auto-detects input format.

This module deliberately does NOT:
- run hhfilter or any other post-search filter (the mmseqs_local builder filters
  per-DB only through mmseqs `--filter-msa`, itself gated at
  `--filter-min-enable 1000`; hhblits has no equivalent gate, so nothing is
  filtered here). hhfilter runs later, in the merge phase, per DB-kind group,
  and only under `merge.mode: leveled`.
- do any merging / deduplication (`hhblits/build.py:build_a3m_hhblits` merges
  the per-DB a3ms textually — dedup, then hhfilter and head-cap — in its merge
  phase)
"""

import logging
from pathlib import Path

from MSA.local_msa.hhblits.runner import run_hhblits

logger = logging.getLogger(__name__)


def run_hhblits_per_chain(
    input_paths: list[Path],
    db_stem: Path,
    out_dir: Path,
    *,
    log_dir: Path | None = None,
    n_iter: int = 3,
    evalue: float = 0.001,
    threads: int = 16,
    hhblits: Path | None = None,
) -> list[Path]:
    """Run hhblits once per chain against `db_stem`, writing
    `out_dir/<cid>.a3m` for cid in [0, len(input_paths)).

    Args:
        input_paths: per-chain queries. Length == n_unique. Each element
            is either a FASTA (Stage A) or an a3m from an upstream stage
            (Stage C iterative). hhblits auto-detects.
        db_stem: hhblits DB path prefix (`<stem>_cs219.ffindex` etc must
            exist).
        out_dir: output directory; created if missing. Per-chain outputs
            named `<cid>.a3m` (and `<cid>.hhr` for the report).
        log_dir: per-chain log dir. Default: `out_dir/_logs`.
        n_iter: hhblits `-n`. Default 3 (used for both Stage A and C).
        evalue: hhblits `-e`. Default 0.001.
        threads: hhblits `-cpu`.
        hhblits: binary override.

    Returns:
        List of output a3m paths in cid order.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if log_dir is None:
        log_dir = out_dir / "_logs"
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    db_stem = Path(db_stem)
    cs219 = Path(f"{db_stem}_cs219.ffindex")
    if not cs219.is_file():
        raise FileNotFoundError(
            f"hhblits DB missing: {cs219} (db_stem={db_stem})"
        )

    out_paths: list[Path] = []
    for cid, in_path in enumerate(input_paths):
        in_path = Path(in_path)
        if not in_path.is_file():
            raise FileNotFoundError(f"chain {cid} input not found: {in_path}")
        out_a3m = out_dir / f"{cid}.a3m"
        out_hhr = out_dir / f"{cid}.hhr"
        run_hhblits(
            input_path=in_path,
            output_a3m=out_a3m,
            db_stem=db_stem,
            output_hhr=out_hhr,
            n_iter=n_iter,
            evalue=evalue,
            threads=threads,
            log_path=log_dir / f"{cid}.log",
            hhblits=hhblits,
        )
        out_paths.append(out_a3m)
        logger.info("hhblits chain %d → %s", cid, out_a3m)
    return out_paths
