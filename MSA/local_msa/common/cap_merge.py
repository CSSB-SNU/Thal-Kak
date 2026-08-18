"""Per-DB hit caps + total-budget cap-walk (pure-Python, record-based).

Helpers operating on a3m RECORDS (lines starting with '>'), with the query
as record 0 ALWAYS counted:

  - `count_a3m_records` / `head_a3m_records`  record-count + head-truncate
    (used for the paired head-N cap; query row included).
  - `collect_per_db_a3m`  resolves the priority-ordered list of
    `(db_key, <cid>.a3m)` for one chain, intersecting the cap priority with
    the engine's effective db set.
  - `cap_merge_chain`  seeds the query once, then walks each per-DB a3m in
    priority order: head-capped per kind, raw_seq-deduped vs everything
    seen, stopping at the per-chain total budget.

Reuses `_iter_records` (header/body record iterator) and `raw_seq_key`
(literal aa-seq dedup key) from `common/dedup.py`.
"""

from __future__ import annotations

from pathlib import Path

from MSA.local_msa.common.dedup import _iter_records, raw_seq_key


def count_a3m_records(text: str) -> int:
    """Number of a3m records ('>' header lines) in `text` (query included)."""
    return sum(1 for ln in text.splitlines() if ln.startswith(">"))


def head_a3m_records(text: str, n: int) -> str:
    """Keep the first `n` records (query row 0 included), preserving the
    original lines (multi-line bodies intact). Records past `n` are dropped."""
    out: list[str] = []
    seen = 0
    for ln in text.splitlines():
        if ln.startswith(">"):
            seen += 1
            if seen > n:
                break
        out.append(ln)
    return "\n".join(out) + ("\n" if out else "")


def collect_per_db_a3m(
    cid: int,
    *,
    effective_dbs: list[str],
    per_db_dir_of: dict[str, Path],
    priority: list[str],
) -> list[tuple[str, Path]]:
    """Resolve the priority-ordered per-DB a3m paths for one chain.

    Walks `priority`; for each key that IS in `effective_dbs`, the path is
    `per_db_dir_of[key] / f"{cid}.a3m"`. A priority key NOT in
    `effective_dbs` is silently skipped. An effective db whose `<cid>.a3m`
    is missing raises RuntimeError — that means upstream search produced no
    per-chain a3m for a DB we were told to use, which is not something to
    paper over.

    Returns `(db_key, path)` pairs in priority order. Caller must have
    already passed `validate_caps`, which guarantees every effective db is
    in `priority` (so none is silently dropped here).
    """
    effective = set(effective_dbs)
    out: list[tuple[str, Path]] = []
    for key in priority:
        if key not in effective:
            continue
        path = per_db_dir_of[key] / f"{cid}.a3m"
        if not path.is_file():
            raise RuntimeError(
                f"chain {cid} missing per-DB a3m for effective db {key!r}: "
                f"{path} (upstream search must have failed silently?)"
            )
        out.append((key, path))
    return out


def cap_merge_chain(
    per_db_a3m: list[tuple[str, Path]],
    *,
    kinds: dict[str, str],
    per_kind_cap: dict[str, int],
    total_budget: int,
    dedup_mode: str = "raw_seq",
) -> str:
    """Priority-walk per-DB a3ms into one capped, deduped a3m text.

    Algorithm (all counts are RECORDS, query included):
      (a) read each input's first record; if their query bodies differ ->
          ValueError.
      (b) seed the query as record 0 (1 record).
      (c) for each (db_key, path) in `per_db_a3m` order: take the head of
          `per_kind_cap[kinds[db_key]]` records INCLUDING the query row from
          that file (query + up to cap-1 hit records), appending its hit
          records while applying raw_seq dedup vs everything already
          emitted, stopping as soon as the accumulated output reaches
          `total_budget` records.
      (d) return the merged a3m text.

    Args:
        per_db_a3m: `(db_key, path)` in priority order (from
            `collect_per_db_a3m`). Empty -> empty output.
        kinds: db_key -> kind ('uniref' | 'env').
        per_kind_cap: kind -> head record cap (query included).
        total_budget: per-chain total record cap (query included);
            == total_max - N where N is the paired record count.
        dedup_mode: only "raw_seq" is supported.

    Returns:
        Merged a3m text (records joined "{header}\\n{body}\\n"). Empty string
        when `per_db_a3m` is empty.
    """
    if dedup_mode != "raw_seq":
        raise ValueError(
            f"cap_merge_chain supports dedup_mode='raw_seq' only, "
            f"got {dedup_mode!r}"
        )
    if not per_db_a3m:
        return ""

    # (a) read all inputs once; verify a shared query body.
    records_by_db: list[tuple[str, list[tuple[str, str]]]] = []
    query_header: str | None = None
    query_body: str | None = None
    for db_key, path in per_db_a3m:
        records = list(_iter_records(Path(path).read_text()))
        if not records:
            raise RuntimeError(
                f"per-DB a3m for {db_key!r} is empty: {path} "
                f"(expected at least the query record)"
            )
        if query_body is None:
            query_header, query_body = records[0]
        elif records[0][1] != query_body:
            raise ValueError(
                f"query mismatch in {path}: first-record body differs "
                f"from earlier input"
            )
        records_by_db.append((db_key, records))

    # (b) seed the query once.
    seen: set[str] = {raw_seq_key(query_body)}
    out: list[str] = [f"{query_header}\n{query_body}\n"]
    n_out = 1
    if n_out >= total_budget:
        return "".join(out)

    # (c) per-DB head cap (query incl) + raw_seq dedup + total stop.
    for db_key, records in records_by_db:
        cap = per_kind_cap[kinds[db_key]]
        taken_from_db = 1  # the query row counts against this DB's head cap
        for header, body in records[1:]:
            if taken_from_db >= cap:
                break
            taken_from_db += 1
            key = raw_seq_key(body)
            if key in seen:
                continue
            seen.add(key)
            out.append(f"{header}\n{body}\n")
            n_out += 1
            if n_out >= total_budget:
                return "".join(out)

    return "".join(out)
