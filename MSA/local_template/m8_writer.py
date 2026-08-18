"""Synthesize ColabFold-format pdb70.m8 rows from a query→hit residue mapping.

Downstream consumers (`Structure/script/protenix/process_msa_to_json.py`)
parse 4 m8 columns:
    cols[1]  = "<pdb_id>_<auth_chain_id>"
    cols[6]  = q_start (1-based)
    cols[8]  = h_start (1-based)
    cols[-1] = CIGAR (M/I/D run-length)

The other 9 columns (pident/aln_len/mismatches/gapopen/q_end/h_end/evalue/bits)
are not read but a `len(cols) < 12: continue` guard means we must emit ≥12
columns plus the trailing CIGAR (= 13 columns total).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CigarData:
    q_start: int  # 1-based query position of first M
    q_end: int    # 1-based query position of last M (inclusive)
    h_start: int  # 1-based hit position of first M (in FULL structure_sequence)
    h_end: int    # 1-based hit position of last M (inclusive)
    cigar: str
    n_match: int
    aln_len: int  # M + I + D in CIGAR (= alignment column count)


def synthesize_cigar(q2h: dict[int, int]) -> CigarData | None:
    """Convert a 0-based query→hit mapping into m8 CIGAR + 1-based coordinates.

    `q2h` maps 0-based query indices → 0-based indices into the FULL hit
    `structure_sequence`. It is built by the hmmer engine (`engine_hmmer`)
    from the hmmsearch alignment (the mmseqs engine emits convertalis CIGAR
    directly and does not use this).

    Returns None for an empty mapping.

    Between consecutive M's the algorithm emits all `I`s (query-side
    gap-fill) before any `D`s (template-side gap-fill). The original a3m's
    interleaved order is lost, but `parse_m8_cigar` only counts ops per
    type, so the mapping round-trips exactly.
    """
    if not q2h:
        return None

    pairs = sorted(q2h.items())
    ops: list[tuple[str, int]] = []
    q_prev = pairs[0][0] - 1
    h_prev = pairs[0][1] - 1

    for q, h in pairs:
        dq = q - q_prev - 1
        dh = h - h_prev - 1
        if dq < 0 or dh < 0:
            raise ValueError(
                f"non-monotonic q→h mapping at q={q} h={h} (prev q={q_prev} h={h_prev})"
            )
        if dq > 0:
            ops.append(("I", dq))
        if dh > 0:
            ops.append(("D", dh))
        ops.append(("M", 1))
        q_prev, h_prev = q, h

    # Run-length-merge consecutive ops of the same kind.
    merged: list[tuple[str, int]] = []
    for op, count in ops:
        if merged and merged[-1][0] == op:
            merged[-1] = (op, merged[-1][1] + count)
        else:
            merged.append((op, count))
    cigar = "".join(f"{count}{op}" for op, count in merged)

    n_match = sum(c for op, c in merged if op == "M")
    aln_len = sum(c for op, c in merged)
    return CigarData(
        q_start=pairs[0][0] + 1,
        q_end=pairs[-1][0] + 1,
        h_start=pairs[0][1] + 1,
        h_end=pairs[-1][1] + 1,
        cigar=cigar,
        n_match=n_match,
        aln_len=aln_len,
    )


def build_m8_row(
    query_id: str,
    pdb_id: str,
    auth_chain_id: str,
    cigar_data: CigarData,
    *,
    evalue: float = 1e-10,
) -> str:
    """Build a single 13-column m8 TSV row.

    Columns: query, target, pident, aln_len, mismatches, gapopen,
             q_start, q_end, t_start, t_end, evalue, bits, cigar.

    Dummies for unread columns: mismatches=0, gapopen=0, bits=0.0. The hmmer
    engine passes a real e-value (parsed from the hmmsearch domtbl); the
    `1e-10` default stands in when the caller has none, and the downstream
    readers do not consume the column either way.
    """
    target = f"{pdb_id}_{auth_chain_id}"
    pident = 100.0 * cigar_data.n_match / cigar_data.aln_len if cigar_data.aln_len else 0.0
    return "\t".join(
        [
            query_id,
            target,
            f"{pident:.2f}",
            str(cigar_data.aln_len),
            "0",
            "0",
            str(cigar_data.q_start),
            str(cigar_data.q_end),
            str(cigar_data.h_start),
            str(cigar_data.h_end),
            f"{evalue:.2e}",
            "0.0",
            cigar_data.cigar,
        ]
    )


def parse_m8_cigar(query_start: int, hit_start: int, cigar: str) -> dict[int, int]:
    """1-based query → 1-based hit mapping, mirroring the downstream parser.

    Vendored from `Structure/script/protenix/process_msa_to_json.py` so we
    can round-trip-test our synthesis without importing through subpackages.
    """
    mapping: dict[int, int] = {}
    q, h = query_start, hit_start
    for length, op in re.findall(r"(\d+)([MIDNSHP=X])", cigar):
        length = int(length)
        if op in ("M", "=", "X"):
            for _ in range(length):
                mapping[q] = h
                q += 1
                h += 1
        elif op == "I":
            q += length
        elif op == "D":
            h += length
    return mapping
