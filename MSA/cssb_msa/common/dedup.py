"""Merge-time a3m dedup for the cssb MSA build (`dedup_a3m_text`).

`dedup_a3m_text(text, mode)` collapses repeated/duplicate rows in a merged
(multi-DB) per-chain a3m; first occurrence wins (so with UniRef-first merge
order the UniRef copy of a shared hit survives).

Scope — UNPAIRED only: this runs on the per-chain unpaired block (via
`leveled_merge.leveled_merge_texts` or `cap_merge.cap_merge_chain`), where
per-chain independence is correct (`assemble._pad_sequences` gap-pads each
chain separately). The multimer PAIRED block is never row-deduped per chain
(that would desync positional pairing): cross-source paired merge dedups on
the whole-row cross-chain key (`combined.merge.merge_paired_texts`), and the
builders + `assemble._pair_sequences` hard-assert equal paired row counts.

Modes:
  - ``none``      → strip only the repeated query rows the per-DB concat leaves
  - ``raw_seq``   → dedup on the literal aa-seq (default; collapses the same
                    sequence regardless of alignment)
  - ``dedup_key`` → dedup on the AF3/hhblits insert-stripped key

On ``dedup_key`` the result is byte-equivalent to AF3's own multi-MSA
deduplication, which keys on the same insert-stripped form.

Also re-exports `iter_a3m_records` / `a3m_dedup_key` / `A3M_GAP_BYTE` for the
in-tree plotting/Neff consumers (`cssb_msa/plot/neff.py`).
"""

import re
import string
from collections.abc import Iterable

_DELETE_LOWERCASE = str.maketrans("", "", string.ascii_lowercase)
_GAP_BYTE = ord("-")


def _dedup_key(body: str) -> str:
    return body.translate(_DELETE_LOWERCASE)


def _iter_records(text: str) -> Iterable[tuple[str, str]]:
    header: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if header is not None:
                yield header, "".join(buf)
            header, buf = line, []
        elif line:
            buf.append(line)
    if header is not None:
        yield header, "".join(buf)


_NONALPHA = re.compile(r"[^A-Za-z]")


def raw_seq_key(body: str) -> str:
    """Literal amino-acid sequence key: strip ALL gaps and uppercase the
    insertions. Collapses the same DB entry even when aligned differently
    (the inverse of `_dedup_key`, which keeps gaps but drops insertions —
    the two are non-nested equivalence relations).

    Assumes a PROTEIN-alphabet a3m (the mmseqs_cssb invariant; RNA targets
    are routed to skip upstream).
    """
    return _NONALPHA.sub("", body).upper()


def dedup_a3m_text(text: str, *, mode: str = "raw_seq") -> tuple[str, int, int]:
    """Dedup a merged (multi-DB) a3m text. First record must be the query.

    Modes:
      - ``none``:      drop only records whose FULL body byte-equals the
                       query body — strips the repeated query row each
                       per-DB a3m contributes at its boundary; no
                       hit-level dedup (colabfold-canonical).
      - ``raw_seq``:   first-occurrence dedup on ``raw_seq_key`` (literal
                       aa-seq — collapses the same sequence regardless of
                       alignment). Subsumes ``none``. **Default.**
      - ``dedup_key``: first-occurrence dedup on ``_dedup_key`` (insert-
                       stripped body — the AF3 / hhblits key, collapses the
                       same aligned representation).

    Records are processed in input order, so with UniRef-first merge order
    the UniRef copy of a shared hit wins. Returns
    ``(cleaned_text, n_records_in, n_records_out)``; empty input is returned
    unchanged.
    """
    records = list(_iter_records(text))
    n_in = len(records)
    if n_in == 0:
        return text, 0, 0
    query_body = records[0][1]

    if mode == "none":
        out = [records[0]]
        out += [(h, b) for h, b in records[1:] if b != query_body]
    elif mode in ("raw_seq", "dedup_key"):
        keyfn = raw_seq_key if mode == "raw_seq" else _dedup_key
        seen: set[str] = set()
        out = []
        for h, b in records:
            k = keyfn(b)
            if k in seen:
                continue
            seen.add(k)
            out.append((h, b))
    else:
        raise ValueError(
            f"unknown dedup mode {mode!r}; expected none|raw_seq|dedup_key"
        )

    n_out = len(out)
    cleaned = "\n".join(f"{h}\n{b}" for h, b in out) + "\n"
    return cleaned, n_in, n_out


# Public re-exports for in-tree consumers (plotting/Neff).
iter_a3m_records = _iter_records
a3m_dedup_key = _dedup_key
A3M_GAP_BYTE = _GAP_BYTE
