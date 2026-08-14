#!/usr/bin/env python3
"""Convert BioMolDB merged.fasta into AF3-compatible protein-only seqres.

One-off: re-run only when the BioMolDB snapshot is replaced.

Pipeline
--------
Input  : a BioMolDB snapshot's `merged.fasta` (`--in`), ~6.2M records
         (sample types: polypeptide(L), polypeptide(D),
         polyribonucleotide, polydeoxyribonucleotide, branched, non-polymer, ...).

Filter : keep only `polypeptide(L)` (~1.64M records).

Dedup  : key = (pdb_lower, auth_chain, sequence). BioMolDB stores the same
         author chain repeatedly across `(assembly_id, model_id, alt_id)`
         variants — collapse them so hmmsearch doesn't see the same chain
         multiple times (which would inflate hit counts and template files).

Output : pdb_seqres_protein.fasta with header
             >{pdb_lower}_{auth_chain} mol:protein length:{N}
         matching the pdb70/hmmsearch hit-description regex (originally from
         AlphaFold3's data/templates.py `_HIT_DESCRIPTION_REGEX`).

Sequence is left as-is; hmmsearch's `--alphabet amino` ignores
non-standard one-letter codes for modified residues.

`--in` and `--out` are both required: the snapshot to read and where the
seqres FASTA goes are deployment choices, and a wrong default here would
silently rebuild the wrong template DB. The engines read the result from
the template snapshot's `fasta/pdb_seqres_protein.fasta`
(`MSA/cssb_template/_common.py:DEFAULT_SEQRES_FASTA`).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterator

PDB_ID_RE = re.compile(r"^[a-z0-9]{4,}$")
AUTH_CHAIN_RE = re.compile(r"^\w+$")


def parse_header(header: str) -> tuple[str, str] | None:
    """Parse `>4GMH_A_A | polypeptide(L) | Auth:A`.

    Returns (pdb_lower, auth_chain) or None when the record is not
    `polypeptide(L)` or the header schema doesn't match expectation.
    """
    fields = [s.strip() for s in header.lstrip(">").split("|")]
    if len(fields) < 3:
        return None
    if fields[1] != "polypeptide(L)":
        return None

    # `4GMH_A_A` → first underscore-delimited token is the PDB ID.
    pdb_token = fields[0].split()[0]
    pdb = pdb_token.split("_", 1)[0].lower()
    if not PDB_ID_RE.match(pdb):
        return None

    auth_field = fields[2]
    if not auth_field.startswith("Auth:"):
        return None
    auth = auth_field[len("Auth:") :].strip()
    if not auth or not AUTH_CHAIN_RE.match(auth):
        return None

    return pdb, auth


def iter_fasta(path: Path) -> Iterator[tuple[str, str]]:
    """Yield (header_line, sequence) for each FASTA record."""
    header: str | None = None
    parts: list[str] = []
    with path.open() as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(parts)
                header = line
                parts = []
            elif line:
                parts.append(line.strip())
        if header is not None:
            yield header, "".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build AF3-compatible protein-only seqres from BioMolDB merged.fasta",
    )
    ap.add_argument(
        "--in",
        dest="in_fasta",
        type=Path,
        required=True,
        help="Input merged.fasta from the BioMolDB snapshot",
    )
    ap.add_argument(
        "--out",
        dest="out_fasta",
        type=Path,
        required=True,
        help="Output protein-only seqres fasta "
             "(the engines read <template root>/<snapshot>/fasta/pdb_seqres_protein.fasta;"
             " see db_paths.yaml)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, stop after reading this many input records (for smoke runs)",
    )
    args = ap.parse_args(argv)

    in_path: Path = args.in_fasta
    out_path: Path = args.out_fasta

    if not in_path.is_file():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[tuple[str, str, str]] = set()
    n_total = 0
    n_polyp_l = 0
    n_kept = 0
    n_dropped_dup = 0
    n_dropped_noparse = 0
    n_dropped_empty = 0
    pdb_counter: Counter[str] = Counter()
    length_sum = 0
    length_min = -1
    length_max = 0

    t0 = time.time()
    with out_path.open("w") as fout:
        for header, seq in iter_fasta(in_path):
            n_total += 1
            if args.limit and n_total > args.limit:
                break

            parsed = parse_header(header)
            if parsed is None:
                if "polypeptide(L)" in header:
                    n_dropped_noparse += 1
                continue
            n_polyp_l += 1

            if not seq:
                n_dropped_empty += 1
                continue

            pdb, auth = parsed
            key = (pdb, auth, seq)
            if key in seen:
                n_dropped_dup += 1
                continue
            seen.add(key)

            n_kept += 1
            pdb_counter[pdb] += 1
            length_sum += len(seq)
            if length_min < 0 or len(seq) < length_min:
                length_min = len(seq)
            if len(seq) > length_max:
                length_max = len(seq)

            fout.write(f">{pdb}_{auth} mol:protein length:{len(seq)}\n")
            fout.write(seq + "\n")

    dt = time.time() - t0
    avg_len = length_sum / n_kept if n_kept else 0.0
    if length_min < 0:
        length_min = 0

    print(f"Input file                : {in_path}")
    print(f"Output file               : {out_path}")
    print(f"Wall                      : {dt:.1f}s")
    print(f"Total fasta records read  : {n_total}")
    print(f"polypeptide(L) records    : {n_polyp_l}")
    print(f"  parse failures          : {n_dropped_noparse}")
    print(f"  empty sequence          : {n_dropped_empty}")
    print(f"  duplicate (pdb,auth,seq): {n_dropped_dup}")
    print(f"Kept (after dedup)        : {n_kept}")
    print(f"Unique PDB IDs            : {len(pdb_counter)}")
    print(f"Sequence length min/avg/max: {length_min} / {avg_len:.1f} / {length_max}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
