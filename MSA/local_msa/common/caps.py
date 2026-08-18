"""Per-DB hit caps + total MSA depth budget — config normalization + validation.

The local MSA engines (`mmseqs_local`, `hhblits_local`) cap each per-DB
per-chain a3m to a kind-based record budget (uniref/env), build the paired
a3m to a head-N record cap, then priority-walk the per-DB a3ms into one
merged a3m bounded by a per-chain total record budget. All counts are a3m
RECORDS (lines starting with '>'), INCLUDING the query row (record 0).

`normalize_caps(cfg)` is the single source of truth shared across the
PROCESS BOUNDARY: the parent (`MSA/msa_generation.py`, skip-key), the
subprocess (`MSA/local_msa/__main__.py`, validation + method_log), and the
builders (`mmseqs/build.py`, `hhblits/build.py`). There is NO `enable`
flag — caps are always applied; an absent `caps:` block yields the
defaults below.

`validate_caps(norm, effective_dbs, kinds, merge_mode)` fail-fasts on a
malformed caps config or a DB the priority-walk would silently drop. It runs
in the subprocess against the config's db set. **Which fields it checks
depends on `merge_mode`** — see `_LIVE_CAPS`.
"""

from __future__ import annotations

from MSA.local_msa.common.db_registry import CAP_PRIORITY, DEFAULT_REGISTRY

# Defaults. Records, query included.
_DEFAULT_PER_KIND: dict[str, int] = {"uniref": 10000, "env": 5000}
_DEFAULT_PAIRED_MAX: int = 8192
_DEFAULT_TOTAL_MAX: int = 16384

# Which caps fields the merge actually reads, per `merge.mode`. `leveled` bounds
# depth with a staged hhfilter over DB-kind groups instead of a per-DB head-N, so
# it never reads `per_kind` or `priority`; `cap_walk` reads all four.
#
# Two consumers read this table: `validate_caps` below, which must not reject a
# config over a field its merge ignores, and the re-run skip key in
# `MSA/msa_generation.py`, which must not compare one.
_LIVE_CAPS: dict[str, tuple[str, ...]] = {
    "leveled": ("paired_max", "total_max"),
    "cap_walk": ("per_kind", "paired_max", "total_max", "priority"),
}


def live_caps(norm: dict, merge_mode: str) -> dict:
    """The subset of `norm` that changes the a3m under `merge_mode`.

    Use this — not the whole dict — whenever caps are compared for equality.
    Reads with `.get` so a method_log written by an older build compares as a
    mismatch (→ rebuild) instead of raising.

    Raises:
        ValueError on an unknown `merge_mode`.
    """
    try:
        fields = _LIVE_CAPS[merge_mode]
    except KeyError:
        raise ValueError(
            f"unknown merge mode {merge_mode!r}; expected one of {sorted(_LIVE_CAPS)}"
        ) from None
    return {k: norm.get(k) for k in fields}


def normalize_caps(cfg: dict) -> dict:
    """Normalize the `caps:` block of an MSA config into a complete dict.

    Returns
        ``{"per_kind": {"uniref": int, "env": int},
           "paired_max": int, "total_max": int, "priority": list[str]}``.

    No ``enable`` key — caps are always applied. If ``cfg`` has no ``caps``
    block, every field defaults (uniref 10000, env 5000, paired_max 8192,
    total_max 16384, priority = ``CAP_PRIORITY``). If present, missing keys
    are filled with those defaults; a present ``per_kind`` fills only its
    own missing sub-keys.

    Shared verbatim by the parent (skip-key), the subprocess (validation),
    and the builders (method_log) so the same config maps to the same dict
    across the process boundary. Output is YAML-round-trip stable
    (lists/ints/str) so ``==`` compares cleanly against a method_log reload.
    """
    caps = cfg.get("caps") or {}

    per_kind_in = caps.get("per_kind") or {}
    per_kind = {
        "uniref": int(per_kind_in.get("uniref", _DEFAULT_PER_KIND["uniref"])),
        "env": int(per_kind_in.get("env", _DEFAULT_PER_KIND["env"])),
    }

    priority = caps.get("priority")
    priority = list(CAP_PRIORITY) if priority is None else list(priority)

    return {
        "per_kind": per_kind,
        "paired_max": int(caps.get("paired_max", _DEFAULT_PAIRED_MAX)),
        "total_max": int(caps.get("total_max", _DEFAULT_TOTAL_MAX)),
        "priority": priority,
    }


def validate_caps(
    norm: dict,
    effective_dbs: list[str],
    kinds: dict[str, str],
    merge_mode: str,
) -> None:
    """Fail-fast on a malformed normalized caps dict or a droppable DB.

    Only the fields `merge_mode` actually reads are checked (`_LIVE_CAPS`).
    Under `leveled` that means `per_kind` and `priority` go unchecked: the merge
    never reads them, so rejecting a db list over `caps.priority` would refuse a
    config that would have run correctly — and would blame the cap-walk, which is
    not running. Switching back to `cap_walk` re-arms both checks.

    Args:
        norm: a `normalize_caps()` result.
        effective_dbs: the db keys the engine will actually search.
        kinds: kind of each key in `effective_dbs`. The caller builds it,
            because the hhblits engine does not use `DBSpec.kind` (only its
            `dbs[0]` counts as uniref there; see `common/leveled_merge.py`).
        merge_mode: the resolved `merge.mode` ('leveled' | 'cap_walk').

    Raises:
        ValueError if paired_max < 0; total_max < 1; paired_max >= total_max (so
        total_budget = total_max - N >= 1 and the query can always be seeded);
        any effective db lacks a kind in {"uniref", "env"} (both modes group by
        kind); `merge_mode` is unknown. Under `cap_walk` additionally: any
        per_kind value <= 0; any priority key not in DEFAULT_REGISTRY; priority
        has duplicates; or any effective db is absent from priority.
    """
    live = live_caps(norm, merge_mode)  # also rejects an unknown merge_mode

    if "per_kind" in live:
        for kind, cap in norm["per_kind"].items():
            if cap <= 0:
                raise ValueError(f"caps.per_kind[{kind!r}] must be > 0, got {cap}")

    paired_max = norm["paired_max"]
    total_max = norm["total_max"]
    if paired_max < 0:
        raise ValueError(f"caps.paired_max must be >= 0, got {paired_max}")
    if total_max < 1:
        raise ValueError(f"caps.total_max must be >= 1, got {total_max}")
    if paired_max >= total_max:
        raise ValueError(
            f"caps.paired_max ({paired_max}) must be < caps.total_max "
            f"({total_max}) so the unpaired budget total_max - N >= 1"
        )

    check_priority = "priority" in live
    # Only ever read under check_priority (the loop below short-circuits on it);
    # bound here so it exists on the path where `priority` is not live.
    priority_set = set()
    if check_priority:
        priority = norm["priority"]
        unknown = [k for k in priority if k not in DEFAULT_REGISTRY]
        if unknown:
            raise ValueError(
                f"caps.priority has keys not in DEFAULT_REGISTRY: {unknown}"
            )
        if len(set(priority)) != len(priority):
            dupes = sorted({k for k in priority if priority.count(k) > 1})
            raise ValueError(f"caps.priority has duplicate keys: {dupes}")
        priority_set = set(priority)

    for db in effective_dbs:
        if check_priority and db not in priority_set:
            raise ValueError(
                f"effective db {db!r} is not in caps.priority "
                f"(the cap-walk would silently drop it)"
            )
        # Both modes group by kind — cap_walk to pick a per-kind cap, leveled to
        # build the uniref/env hhfilter groups — so this one is never skipped.
        if kinds.get(db) not in ("uniref", "env"):
            raise ValueError(
                f"effective db {db!r} has no uniref/env kind "
                f"(kind={kinds.get(db)!r}); cannot place it in a merge group"
            )
