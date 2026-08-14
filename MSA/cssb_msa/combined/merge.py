"""Stage-2 cross-source MSA merge for the `mmseqs_hhblits_cssb` mode.

Merges the two source builders' (`mmseqs/build.build_a3m`,
`hhblits/build.build_a3m_hhblits`) per-chain outputs. `merge_paired_*` runs
under both `merge.mode` values; the unpaired half (`merge_unpaired_*`,
re-exported below) is used by the `cap_walk` path only — under the default
`leveled`, `combined/build.py` groups per-DB a3ms across both sources instead
(see `common/leveled_merge.py`).

  - **dedup key** = ``raw_seq_key`` (gaps + insertions stripped, uppercased =
    literal aa-seq; cf. `common/dedup.raw_seq_key`). Two rows collide iff their
    bare amino-acid sequence matches, regardless of gaps/case/alignment.
  - **mmseqs priority**: mmseqs records are emitted BEFORE hhblits records, so
    on a raw_seq collision the hhblits copy is the one dropped (first-occurrence
    wins). The shared query row (record 0 of both sources) is therefore kept
    once, from mmseqs.
  - **re-cap**: after dedup, head-truncate to the per-chain budget (records
    INCLUDING the query row 0). Because mmseqs records come first they are
    preferentially retained; hhblits diversity fills whatever budget remains.

Unpaired budget = ``total_max - N`` (N = this chain's merged paired record
count); paired budget = ``paired_max``. Paired dedup is at the PAIR level (the
cross-chain tuple of per-chain raw_seq keys at a given row), so positional
pairing across chains is preserved.
"""

from __future__ import annotations

from pathlib import Path

from MSA.cssb_msa.common.dedup import raw_seq_key

# The unpaired half lives in `common/merge_unpaired.py` so `common/leveled_merge.py`
# can use it without `common/` importing from `combined/`. Re-exported here for
# callers of `combined.merge.merge_unpaired_*`.
from MSA.cssb_msa.common.merge_unpaired import (  # noqa: F401
    merge_unpaired_file,
    merge_unpaired_text,
    merge_unpaired_texts,
    records as _records,
)


# Paired (multimer; cross-chain positional pairing)
def _records_per_chain(texts: list[str]) -> tuple[list[list[tuple[str, str]]], int]:
    """Parse per-chain paired a3m texts; assert equal record count across
    chains (the positional-pairing invariant the builders guarantee). Returns
    ``(per_chain_records, n_pairs)``; an all-empty source yields n_pairs=0."""
    per_chain = [_records(t) for t in texts]
    counts = {len(r) for r in per_chain}
    if len(counts) > 1:
        raise ValueError(
            f"paired a3m record counts differ across chains: "
            f"{[len(r) for r in per_chain]} — positional pairing broken"
        )
    return per_chain, (counts.pop() if counts else 0)


def merge_paired_texts_multi(
    sources_texts: list[list[str]],
    *,
    paired_max: int,
) -> list[str]:
    """Merge N ordered sources' per-chain paired a3ms → one merged per-chain list.

    Args:
        sources_texts: ordered list of per-source per-chain paired a3m text lists
            ``[[chain0, chain1, ...] (source0), ...]`` — all sources same n_chains;
            within each source every chain has equal record count (positional
            pairing). Earlier source = priority.
        paired_max: cap on the number of paired records (query row 0 incl).

    Returns:
        Per-chain merged a3m texts (n_chains long, equal record count across
        chains). Source 0 pairs first, then each later source's pairs whose
        cross-chain raw_seq tuple is not already present; truncated to
        ``paired_max``.
    """
    if not sources_texts:
        raise ValueError("sources_texts is empty")
    n_chains = len(sources_texts[0])
    for i, texts in enumerate(sources_texts):
        if len(texts) != n_chains:
            raise ValueError(
                f"chain count mismatch: source0={n_chains} source{i}={len(texts)}"
            )
    if paired_max < 1:
        raise ValueError(f"paired_max must be >= 1, got {paired_max}")

    per_source = [_records_per_chain(texts) for texts in sources_texts]

    def pair_key(per_chain: list[list[tuple[str, str]]], k: int) -> tuple[str, ...]:
        return tuple(raw_seq_key(per_chain[c][k][1]) for c in range(n_chains))

    seen: set[tuple[str, ...]] = set()
    selected: list[tuple[list[list[tuple[str, str]]], int]] = []
    for per_chain, n_pairs in per_source:
        if len(selected) >= paired_max:
            break
        for k in range(n_pairs):
            if len(selected) >= paired_max:
                break
            key = pair_key(per_chain, k)
            if key in seen:
                continue
            seen.add(key)
            selected.append((per_chain, k))

    out_texts: list[str] = []
    for c in range(n_chains):
        buf = [f"{per_chain[c][k][0]}\n{per_chain[c][k][1]}\n" for per_chain, k in selected]
        out_texts.append("".join(buf))
    return out_texts


def merge_paired_texts(
    mmseqs_texts: list[str],
    hhblits_texts: list[str],
    *,
    paired_max: int,
) -> list[str]:
    """Two-source (mmseqs-first) wrapper over `merge_paired_texts_multi`."""
    return merge_paired_texts_multi([mmseqs_texts, hhblits_texts], paired_max=paired_max)


def merge_paired_files(
    mmseqs_paths: list[Path],
    hhblits_paths: list[Path],
    out_paths: list[Path],
    *,
    paired_max: int,
) -> int:
    """File wrapper for `merge_paired_texts`. Missing inputs read as empty.
    Returns the merged pair count (records per chain)."""
    mm = [Path(p).read_text() if Path(p).is_file() else "" for p in mmseqs_paths]
    hh = [Path(p).read_text() if Path(p).is_file() else "" for p in hhblits_paths]
    merged = merge_paired_texts(mm, hh, paired_max=paired_max)
    if len(out_paths) != len(merged):
        raise ValueError(
            f"out_paths ({len(out_paths)}) != merged chains ({len(merged)})"
        )
    for out_path, text in zip(out_paths, merged):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
    # All chains share one record count (merge_paired_texts enforces positional
    # equality), so the paired record count N is well-defined from chain 0.
    return sum(1 for ln in merged[0].splitlines() if ln.startswith(">")) if merged else 0
