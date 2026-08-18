"""ColabFold-style mmseqs template search against BioMolDB (mmseqs_local path).

The template engine for the **mmseqs_local** pipeline. The hhblits_local path uses
the local-hmmer engine (`engine_hmmer.run_hmmer_for_msa_dir`); engine selection +
dispatch live in `local_template/dispatch.py`.

Recipe (ground truth: ColabFold API `msa.sh`):

    mmseqs search     prof_res_num <pdbDB>       res tmp --db-load-mode M -s 7.5 -a -e 0.1
    mmseqs convertalis prof_res_num <pdbDB[.idx]> res pdb70.m8 --db-load-mode M \
        --format-output query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,cigar

where `prof_res_num` is the persisted UniRef30 profile (`<msa_dir>/_workdir/prof_res`,
kept by `build_a3m` under keep_intermediate=True) with a **numeric** (`101+cid`) header
DB attached, so convertalis col0 comes out 101/102/... per chain. (local's query DB
headers are all `>{target}`, so a plain convertalis would emit the same col0 on every
chain and downstream `parse_m8_top_templates` — which groups on `int(col0)==101+cid` —
would break. We attach numeric headers to a COPY of the profile; the original prof_res
is never mutated.)

Output (consumed by `colab_a3m_to_yaml.split_colab_a3m_write_yaml` via a disk scan):

    <msa_dir>/<target>_env/pdb70.m8                       # 13-col + cigar, col0 = 101+cid
    <msa_dir>/<target>_env/templates_<101+cid>/<pdb>.cif  # gunzipped from BioMolDB cif/raw

The mmseqs and hmmer engines search the same snapshot by different algorithms, so
their hit sets differ by design; neither is a parity replacement for the other.
"""

from __future__ import annotations

import datetime
import logging
import shutil
from pathlib import Path

from MSA.local_msa.mmseqs.runner import createdb, lndb, resolve_mmseqs, run_mmseqs

from ._common import (
    DEFAULT_MAX_HITS,
    DEFAULT_MAX_TEMPLATE_DATE,
    DEFAULT_MMCIF_DIR,
    MsaDirResult,
    TemplateEntry,
    TEMPLATE_ROOT,
    TEMPLATE_SNAPSHOT,
    _alphabet_letters,
    _emit_zero_hits_banner,
    _gunzip_cif,
)

# Reuse the shared release-date loader (initial release date from the sidecar;
# deposition_date fallback) so both engines apply the SAME cutoff.
from .engine_hmmer import DEFAULT_CIF_METADATA, _load_release_dates

logger = logging.getLogger(__name__)

# The mmseqs DB basename follows the snapshot's build convention
# <snapshot>.lower()+"_pdb" (BioMolDB_20260224 -> biomoldb_20260224_pdb); the
# .dbtype check below fails loud on a snapshot that deviates from it.
DEFAULT_PDB_DB = TEMPLATE_ROOT / "mmseqs" / f"{TEMPLATE_SNAPSHOT.lower()}_pdb"
DEFAULT_SENSITIVITY = 7.5
DEFAULT_EVALUE = 0.1

# `mmseqs convertalis --format-output` field order (matches ColabFold msa.sh).
_CONVERTALIS_FORMAT = (
    "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,cigar"
)
# 0-based column indices into that 13-column output.
_COL_QUERY, _COL_TARGET, _COL_EVALUE, _COL_CIGAR = 0, 1, 10, 12
_N_COLS = 13


def _read_fasta_records(path: Path) -> list[str]:
    """Return sequences (gaps/case preserved) of every FASTA record, in file order."""
    seqs: list[str] = []
    cur: list[str] = []
    in_rec = False
    with Path(path).open() as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if line.startswith(">"):
                if in_rec:
                    seqs.append("".join(cur))
                cur = []
                in_rec = True
            elif in_rec and line:
                cur.append(line.strip())
        if in_rec:
            seqs.append("".join(cur))
    return seqs


def _row_evalue(cols: list[str]) -> float:
    try:
        return float(cols[_COL_EVALUE])
    except (ValueError, IndexError):
        return float("inf")


def _date_filter_rows(
    rows: list[list[str]],
    cif_dates: dict[str, datetime.date],
    max_template_date: datetime.date,
) -> list[list[str]]:
    """Drop rows whose hit PDB date (``cif_dates``) is after ``max_template_date``.

    ``cif_dates`` maps pdb_id_lower -> date (release date with deposition fallback;
    see `_load_release_dates`). Hit id is col1's pdb prefix ('8bwl_A' -> '8bwl',
    matched lowercased). An id absent from ``cif_dates`` is KEPT (no date to filter
    on) — same convention as the hmmer engine. Must be applied BEFORE the per-chain
    max_hits cap so date-valid templates are not crowded out by dropped ones.
    Passing an empty ``cif_dates`` makes this a pass-through (no-cutoff fast path).
    """
    kept: list[list[str]] = []
    for cols in rows:
        pdb_id = cols[_COL_TARGET].partition("_")[0].lower()
        rel = cif_dates.get(pdb_id)
        if rel is not None and rel > max_template_date:
            continue
        kept.append(cols)
    return kept


def run_mmseqs_for_msa_dir(
    msa_dir: Path,
    *,
    pdb_db: Path = DEFAULT_PDB_DB,
    mmcif_dir: Path = DEFAULT_MMCIF_DIR,
    max_template_date: datetime.date = DEFAULT_MAX_TEMPLATE_DATE,
    cif_metadata: Path = DEFAULT_CIF_METADATA,
    max_hits: int = DEFAULT_MAX_HITS,
    sensitivity: float = DEFAULT_SENSITIVITY,
    evalue: float = DEFAULT_EVALUE,
    mmseqs: Path | None = None,
) -> MsaDirResult:
    """ColabFold-style mmseqs template search for a `local_msa` mmseqs_local `msa_dir`.

    Reuses the persisted UniRef30 profile at `<msa_dir>/_workdir/prof_res`, searches it
    against `pdb_db` (a BioMolDB seqres mmseqs DB), and writes the ColabFold-format
    `<target>_env/pdb70.m8` + gunzipped per-hit cifs that the data-yaml writer scans.

    Args mirror the sibling `engine_hmmer.run_hmmer_for_msa_dir` where they overlap. `max_template_date`
    applies a release-date cutoff: hits whose PDB initial release date (release-date
    sidecar, deposition_date fallback) is after the cutoff are dropped BEFORE the
    per-chain max_hits cap. Reuses the shared `_load_release_dates` so both local
    engines filter identically. Default (3000-01-01) = no cutoff.
    """
    msa_dir = Path(msa_dir).resolve()
    workdir = msa_dir / "_workdir"
    prof_res = workdir / "prof_res"
    query_fas = workdir / "query.fas"

    # preconditions (fail fast).
    for p in (prof_res, prof_res.with_suffix(".dbtype"), prof_res.with_suffix(".index")):
        if not p.exists():
            raise FileNotFoundError(
                f"mmseqs template search needs {p} — the persisted UniRef30 profile from "
                f"build_a3m (keep_intermediate=True), intact while _workdir is present. "
                f"The hhblits_local path produces no prof_res; use the hmmer engine there."
            )
    if not (workdir / "prof_res_h").exists():
        raise FileNotFoundError(f"mmseqs template search needs {workdir/'prof_res_h'} (profile header DB)")
    if not query_fas.is_file():
        raise FileNotFoundError(f"missing {query_fas} (build_a3m Phase 1 writes the query fasta)")
    if not Path(str(pdb_db) + ".dbtype").exists():
        raise FileNotFoundError(
            f"pdb mmseqs DB not found at {pdb_db}.dbtype — it is part of the "
            f"template DB snapshot; reinstall it, or rebuild with `mmseqs createdb` "
            f"[+ `createindex`] on the snapshot's seqres FASTA."
        )

    # Release-date cutoff: drop hits RELEASED after max_template_date,
    # using initial release dates (sidecar) with deposition_date fallback, via the
    # shared _load_release_dates. Skip loading entirely when no cutoff is set.
    apply_date_filter = max_template_date != DEFAULT_MAX_TEMPLATE_DATE
    cif_dates = _load_release_dates(cif_metadata=cif_metadata) if apply_date_filter else {}
    if apply_date_filter:
        logger.info(
            "mmseqs template search: applying release-date cutoff %s "
            "(release date via release_dates.tsv; deposition_date fallback).",
            max_template_date.isoformat(),
        )

    # discover master a3m + parse ColabFold-complex header (target + cardinality).
    candidates = [
        p
        for p in msa_dir.glob("*.a3m")
        if "_paired_msa_chains_" not in p.name and "_unpaired_msa_chains_" not in p.name
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one master a3m under {msa_dir}, found {candidates}")
    master_a3m = candidates[0]
    target = master_a3m.stem
    with master_a3m.open() as fh:
        header = fh.readline().rstrip("\r\n")
    if not header.startswith("#"):
        raise RuntimeError(f"master a3m {master_a3m} missing ColabFold-complex header: {header!r}")
    parts = header[1:].split("\t")
    chain_lengths = [int(x) for x in parts[0].split(",")]
    cardinality = (
        [int(x) for x in parts[1].split(",")] if len(parts) > 1 else [1] * len(chain_lengths)
    )
    n_unique = len(chain_lengths)

    # per-cid full alphabet-letter slice (e.g. A2B2 -> {0:[A,B], 1:[C,D]}).
    flat_letters = _alphabet_letters(sum(cardinality))
    chain_letters: dict[int, list[str]] = {}
    cursor = 0
    for cid, count in enumerate(cardinality):
        chain_letters[cid] = flat_letters[cursor : cursor + count]
        cursor += count

    # unique query seqs in cid order (record order of query.fas == prof_res entry order).
    seqs = _read_fasta_records(query_fas)
    if len(seqs) != n_unique:
        raise RuntimeError(
            f"query.fas record count {len(seqs)} != unique-chain count {n_unique} "
            f"(master={master_a3m})"
        )

    out_dir = msa_dir / f"{target}_env"
    out_dir.mkdir(parents=True, exist_ok=True)
    # clean slate: drop prior template artifacts (from an earlier mmseqs run)
    # so a re-run leaves no orphan cifs alongside the fresh output.
    for old in out_dir.glob("templates_*"):
        shutil.rmtree(old, ignore_errors=True)
    (out_dir / "pdb70.m8").unlink(missing_ok=True)
    scratch = out_dir / "_mmseqs_scratch"
    if scratch.exists():
        shutil.rmtree(scratch)
    (scratch / "tmp").mkdir(parents=True, exist_ok=True)
    log_dir = scratch / "logs"
    mmseqs_bin = resolve_mmseqs(mmseqs)

    # (A) numeric-header query DB: >101, >102, ... in cid order.
    numq_fa = scratch / "numq.fasta"
    numq_fa.write_text("".join(f">{101 + i}\n{seq}\n" for i, seq in enumerate(seqs)))
    numq = scratch / "numq"
    createdb(numq_fa, numq, log_path=log_dir / "createdb_numq.log", mmseqs=mmseqs_bin)

    # prof_res_num = prof_res PROFILE DATA (copied) + numeric header DB (lndb). prof_res untouched.
    prof_num = scratch / "prof_res_num"
    shutil.copy(prof_res, prof_num)
    shutil.copy(prof_res.with_suffix(".index"), prof_num.with_suffix(".index"))
    shutil.copy(prof_res.with_suffix(".dbtype"), prof_num.with_suffix(".dbtype"))
    lndb(
        Path(str(numq) + "_h"),
        Path(str(prof_num) + "_h"),
        log_path=log_dir / "lndb_numh.log",
        mmseqs=mmseqs_bin,
    )
    numq_lookup = Path(str(numq) + ".lookup")
    if numq_lookup.exists():
        shutil.copy(numq_lookup, Path(str(prof_num) + ".lookup"))

    # db-load-mode: 2 (mmap, needs precomputed .idx) if present, else 1 (base DB).
    if Path(str(pdb_db) + ".idx").exists() or Path(str(pdb_db) + ".idx.dbtype").exists():
        dbload, conv_db = "2", Path(str(pdb_db) + ".idx")
    else:
        dbload, conv_db = "1", pdb_db

    res = scratch / "res_pdb"
    run_mmseqs(
        [
            "search", str(prof_num), str(pdb_db), str(res), str(scratch / "tmp"),
            "--db-load-mode", dbload, "-s", str(sensitivity), "-a", "-e", str(evalue),
        ],
        log_path=log_dir / "search.log",
        mmseqs=mmseqs_bin,
    )
    raw_m8 = scratch / "pdb70_raw.m8"
    run_mmseqs(
        [
            "convertalis", str(prof_num), str(conv_db), str(res), str(raw_m8),
            "--db-load-mode", dbload, "--format-output", _CONVERTALIS_FORMAT,
        ],
        log_path=log_dir / "convertalis.log",
        mmseqs=mmseqs_bin,
    )

    # group raw m8 rows by col0 (= 101+cid), evalue-sort, cap at max_hits.
    rows_by_qid: dict[int, list[list[str]]] = {}
    with raw_m8.open() as fh:
        for line in fh:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < _N_COLS:
                continue
            try:
                qid = int(cols[_COL_QUERY])
            except ValueError:
                continue
            rows_by_qid.setdefault(qid, []).append(cols)

    final_lines: list[str] = []
    template_entries: list[TemplateEntry] = []
    mmcif_root = Path(mmcif_dir).resolve()

    for cid in range(n_unique):
        qid = 101 + cid
        letters = chain_letters[cid]
        ranked = _date_filter_rows(
            sorted(rows_by_qid.get(qid, []), key=_row_evalue),
            cif_dates,
            max_template_date,
        )
        rows = ranked[:max_hits]
        if not rows:
            _emit_zero_hits_banner(cid, letters, query_fas)
            continue
        template_dir = out_dir / f"templates_{qid}"
        template_dir.mkdir(parents=True, exist_ok=True)
        seen_pdbs: set[str] = set()
        for cols in rows:
            pdb_id, _, auth = cols[_COL_TARGET].partition("_")  # "8bwl_A" -> ("8bwl", "_", "A")
            if not pdb_id or not auth:
                continue
            cif_dst = template_dir / f"{pdb_id}.cif"
            if pdb_id not in seen_pdbs:
                # Fails loud on a missing cif (see `_gunzip_cif`), same as the
                # hmmer engine — an incomplete snapshot must not degrade into a
                # quietly shorter template list.
                _gunzip_cif(mmcif_root / pdb_id[1:3] / f"{pdb_id}.cif.gz", cif_dst)
                seen_pdbs.add(pdb_id)
            final_lines.append("\t".join(cols))
            template_entries.append(
                TemplateEntry(path=cif_dst, chain_template=[auth], chain_query=list(letters))
            )

    m8_path = out_dir / "pdb70.m8"
    m8_path.write_text(("\n".join(final_lines) + "\n") if final_lines else "")

    # scratch hygiene — never touch prof_res*; drop all mmseqs intermediates.
    shutil.rmtree(scratch, ignore_errors=True)

    logger.info(
        "mmseqs template search: %d template rows (%d of %d chains with raw hits) -> %s",
        len(template_entries), len(rows_by_qid), n_unique, m8_path,
    )
    return MsaDirResult(
        out_dir=out_dir,
        m8_path=m8_path,
        template_entries=template_entries,
        per_chain=[],  # single search (not per-cid); kept empty for interface symmetry
        chain_letters=chain_letters,
    )


def main(argv: list[str] | None = None) -> int:
    """Standalone CLI: `python -m MSA.local_template.engine_mmseqs <msa_dir> [...]`."""
    import argparse

    ap = argparse.ArgumentParser(description="ColabFold-style mmseqs template search vs BioMolDB")
    ap.add_argument("msa_dir", type=Path, help="local_msa msa_dir (contains _workdir/prof_res)")
    ap.add_argument("--pdb-db", type=Path, default=DEFAULT_PDB_DB)
    ap.add_argument("--mmcif-dir", type=Path, default=DEFAULT_MMCIF_DIR)
    ap.add_argument("--max-hits", type=int, default=DEFAULT_MAX_HITS)
    ap.add_argument(
        "--max-template-date",
        type=datetime.date.fromisoformat,
        default=DEFAULT_MAX_TEMPLATE_DATE,
        help=(
            "ISO date; templates released after this are filtered out "
            "(release_dates.tsv when available, deposition_date fallback; default: no cutoff)"
        ),
    )
    ap.add_argument("--cif-metadata", type=Path, default=DEFAULT_CIF_METADATA)
    ap.add_argument("--sensitivity", type=float, default=DEFAULT_SENSITIVITY)
    ap.add_argument("--evalue", type=float, default=DEFAULT_EVALUE)
    args = ap.parse_args(argv)

    result = run_mmseqs_for_msa_dir(
        msa_dir=args.msa_dir,
        pdb_db=args.pdb_db,
        mmcif_dir=args.mmcif_dir,
        max_template_date=args.max_template_date,
        cif_metadata=args.cif_metadata,
        max_hits=args.max_hits,
        sensitivity=args.sensitivity,
        evalue=args.evalue,
    )
    print(f"out_dir  : {result.out_dir}")
    print(f"pdb70.m8 : {result.m8_path}")
    print(f"entries  : {len(result.template_entries)}")
    for cid, letters in result.chain_letters.items():
        n = sum(1 for e in result.template_entries if e.chain_query == letters)
        print(f"  chain {101 + cid} (query {letters}): {n} template rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
