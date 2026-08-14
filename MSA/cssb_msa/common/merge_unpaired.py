"""Ordered-source unpaired a3m merge (shared by every builder).

Lives in `common/` rather than `combined/` because `common/leveled_merge.py`
needs it to stack the uniref group on top of the env group, and `common/` must
not import from `combined/`. `combined/merge.py` re-exports these names.

Semantics: records are appended in SOURCE ORDER, so an earlier source's copy of a shared
sequence wins; the dedup key is ``raw_seq_key`` (gaps stripped, insertions
uppercased = the literal amino-acid sequence), so two rows collide iff their bare
sequence matches regardless of alignment; the result is head-capped to
``total_budget`` records, query row included.
"""

from __future__ import annotations

from pathlib import Path

from MSA.cssb_msa.common.dedup import _iter_records, raw_seq_key


def records(text: str) -> list[tuple[str, str]]:
    """``[(header, body)]`` for an a3m text."""
    return list(_iter_records(text))


def merge_unpaired_texts(*texts: str, total_budget: int) -> str:
    """Merge N ordered per-chain unpaired a3m texts → one a3m text.

    Records appended in source order (earlier source = priority); raw_seq
    first-occurrence dedup (earlier source's copy of a shared sequence wins);
    head-cap to ``total_budget`` records (query row included), filled from the
    front. Empty sources contribute nothing. Returns the merged a3m text
    ("{header}\\n{body}\\n" per record); caller writes it.
    """
    if total_budget < 1:
        raise ValueError(f"total_budget must be >= 1, got {total_budget}")
    seen: set[str] = set()
    out: list[str] = []
    for text in texts:
        for header, body in records(text):
            if len(out) >= total_budget:
                return "".join(out)
            key = raw_seq_key(body)
            if key in seen:
                continue
            seen.add(key)
            out.append(f"{header}\n{body}\n")
    return "".join(out)


def merge_unpaired_text(mmseqs_text: str, hhblits_text: str, *, total_budget: int) -> str:
    """Two-source (mmseqs-first) wrapper over `merge_unpaired_texts`."""
    return merge_unpaired_texts(mmseqs_text, hhblits_text, total_budget=total_budget)


def merge_unpaired_file(
    mmseqs_path: Path,
    hhblits_path: Path,
    out_path: Path,
    *,
    total_budget: int,
) -> int:
    """File wrapper for `merge_unpaired_text`. Missing inputs read as empty.
    Returns the merged record count."""
    mm = Path(mmseqs_path).read_text() if Path(mmseqs_path).is_file() else ""
    hh = Path(hhblits_path).read_text() if Path(hhblits_path).is_file() else ""
    text = merge_unpaired_text(mm, hh, total_budget=total_budget)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    return sum(1 for ln in text.splitlines() if ln.startswith(">"))
