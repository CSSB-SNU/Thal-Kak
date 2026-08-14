"""Leveled per-chain unpaired merge — the `merge.mode: leveled` default.

The default for all three cssb modes; `cap_merge.cap_merge_chain` is the
alternative (`merge.mode: cap_walk`). Per unique chain, per DB KIND (uniref / env):

    concat the kind's per-DB a3ms in `order`        (earlier DB wins a dedup tie)
    -> raw_seq dedup                               (common/dedup.dedup_a3m_text)
    -> staged hhfilter                             (common/hhfilter.staged_hhfilter)

then stack the filtered groups **uniref on top, env below**, raw_seq dedup across
them, and finally head-cap to `total_cap`.

Why this and not the cap walk: the cap walk head-truncates each DB to
`caps.per_kind` (uniref 10000 / env 5000) BEFORE anything else, which measurably
truncated real depth — on a 188-chain multimer benchmark 22 chains saturated a cap
and 10 ended up shallower than the ColabFold API's. Leveled reads every per-DB hit and lets
hhfilter remove redundancy (>= `id` % identity) and low-coverage rows instead, so
depth is bounded by information content rather than by an arbitrary head-N.
`caps.per_kind` is therefore NOT applied here; `caps.total_max` is kept only as an
outer safety bound, applied AFTER filtering, and `info["capped"]` /
`info["depth_before_cap"]` record whether it actually bit.

The dedup key here is always ``raw_seq`` — the yaml `dedup.mode` knob (mmseqs-only)
is INERT in this mode; see the note in `examples/msa_config.mmseqs.yaml` and the
warning the mmseqs builder logs when a non-default value is set alongside
`leveled`.

Kind assignment is the CALLER's, not the registry's — see the `kind_of` arg. The
hhblits builder must not use `DEFAULT_REGISTRY[db].kind`: only its Phase-2 primary
(`dbs[0]`) is the uniref group there, and uniref30 sitting in Phase 3 is an env DB.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from MSA.cssb_msa.common.cap_merge import (
    collect_per_db_a3m,
    count_a3m_records,
    head_a3m_records,
)
from MSA.cssb_msa.common.dedup import _iter_records, dedup_a3m_text
from MSA.cssb_msa.common.hhfilter import staged_hhfilter
from MSA.cssb_msa.common.merge_unpaired import merge_unpaired_texts

logger = logging.getLogger(__name__)

# Group stacking order: uniref rows first, so on a raw_seq collision between the
# groups the UniRef copy of a shared hit is the one kept.
KIND_ORDER: tuple[str, ...] = ("uniref", "env")

# `merge_unpaired_texts` requires a positive budget; this stands in for "no cap"
# when caps.total_max is not being applied.
NO_CAP = 10 ** 9

DEFAULT_HHFILTER_CFG: dict = {
    "id": 90, "cov_hi": 75, "cov_lo": 50, "depth_thresh": 2000, "min_keep": 100,
}


MERGE_MODES: tuple[str, ...] = ("leveled", "cap_walk")


def resolve_merge_cfg(cfg: dict) -> dict:
    """Normalize a config's ``merge:`` block → ``{"mode", "hhfilter"?}``.

    Single source of truth for three callers that must agree or the reuse key
    breaks: `cssb_msa/__main__.py` (what the builders are told to do),
    `MSA/msa_generation.py` (the skip-key comparison), and the builders'
    `method_log` write. An absent ``merge:`` block resolves to **leveled** with the
    default hhfilter knobs; missing individual hhfilter keys are filled from
    `DEFAULT_HHFILTER_CFG`.
    """
    block = cfg.get("merge") or {}
    mode = block.get("mode", "leveled")
    if mode not in MERGE_MODES:
        raise ValueError(
            f"yaml merge.mode must be one of {list(MERGE_MODES)}, got {mode!r}"
        )
    if mode != "leveled":
        return {"mode": mode}
    hh = dict(DEFAULT_HHFILTER_CFG)
    given = block.get("hhfilter") or {}
    unknown = set(given) - set(DEFAULT_HHFILTER_CFG)
    if unknown:
        raise ValueError(
            f"yaml merge.hhfilter has unknown key(s) {sorted(unknown)}; "
            f"expected {sorted(DEFAULT_HHFILTER_CFG)}"
        )
    hh.update({k: int(v) for k, v in given.items()})
    return {"mode": mode, "hhfilter": hh}


def merge_key_matches(recorded: dict | None, config: dict) -> bool:
    """Is a cached `method_log.merge` reusable for a run configured as `config`?

    Compares only the fields that change the OUTPUT (mode + hhfilter knobs) and
    ignores the diagnostics the builders also record (`per_chain`, `dedup`,
    `uniref_group_db`, …). A cache with no `merge` key at all never matches
    `leveled`, so it is rebuilt.
    """
    rec = recorded or {}
    if rec.get("mode") != config["mode"]:
        return False
    if config["mode"] != "leveled":
        return True
    return rec.get("hhfilter") == config["hhfilter"]


def _assert_shared_query(per_db_a3m: list[tuple[str, Path]]) -> None:
    """Every per-DB a3m for one chain must start with the SAME query record.

    `cap_merge_chain` checks this (`cap_merge.py:123-141`) and the leveled path
    would otherwise just concatenate, so a chain-index mix-up between two search
    phases would pass silently and produce an a3m aligned to the wrong query.
    """
    query: str | None = None
    for db_key, path in per_db_a3m:
        records = list(_iter_records(Path(path).read_text()))
        if not records:
            raise RuntimeError(
                f"per-DB a3m for {db_key!r} is empty: {path} "
                f"(expected at least the query record)"
            )
        if query is None:
            query = records[0][1]
        elif records[0][1] != query:
            raise ValueError(
                f"query mismatch in {path}: first-record body differs from an "
                f"earlier per-DB a3m for this chain"
            )


def leveled_merge_texts(
    texts_by_kind: dict[str, list[str]],
    *,
    scratch: Path,
    hhfilter_cfg: dict | None = None,
    total_cap: int | None = None,
    hhfilter_bin: str | None = None,
) -> tuple[str, dict]:
    """The leveled core, on TEXTS. Returns ``(merged_a3m_text, info)``.

    `texts_by_kind[kind]` is that kind's a3m texts in dedup-priority order — for a
    single engine that is its per-DB files in `MERGE_ORDER`; for the combined mode
    it is the SAME kind from every source, source 0 first. Grouping across sources
    before filtering (rather than filtering each source then merging) is the whole
    point of `leveled`: one filter pass per kind, no source boundary inside a group.

    Per kind: concat -> raw_seq dedup -> staged hhfilter. Then the kinds are stacked
    in `KIND_ORDER` (uniref first), deduped across groups, and head-capped to
    `total_cap` (None = uncapped).
    """
    cfg = dict(DEFAULT_HHFILTER_CFG if hhfilter_cfg is None else hhfilter_cfg)
    missing_cfg = set(DEFAULT_HHFILTER_CFG) - set(cfg)
    if missing_cfg:
        raise KeyError(f"hhfilter_cfg missing keys: {sorted(missing_cfg)}")
    unknown = set(texts_by_kind) - set(KIND_ORDER)
    if unknown:
        raise ValueError(f"unknown kind(s) {sorted(unknown)}; expected {list(KIND_ORDER)}")

    info: dict = {"kinds": {}, "final_depth": 0, "total_cap": total_cap, "capped": False}
    per_kind_text: dict[str, str] = {}
    for kind in KIND_ORDER:
        texts = texts_by_kind.get(kind) or []
        concat = "".join(t if (not t or t.endswith("\n")) else t + "\n" for t in texts)
        if not concat:
            per_kind_text[kind] = ""
            info["kinds"][kind] = {"n_raw": 0, "n_dedup": 0, "n_out": 0, "skipped": True}
            continue
        deduped, n_raw, n_dedup = dedup_a3m_text(concat, mode="raw_seq")
        filt, finfo = staged_hhfilter(
            deduped, Path(scratch) / kind,
            hhfilter=hhfilter_bin,
            id_cut=cfg["id"], cov_hi=cfg["cov_hi"], cov_lo=cfg["cov_lo"],
            depth_thresh=cfg["depth_thresh"], min_keep=cfg["min_keep"],
            tag=kind,
        )
        per_kind_text[kind] = filt
        info["kinds"][kind] = {"n_raw": n_raw, "n_dedup": n_dedup, **finfo}

    # Stack uncapped first, then truncate. Doing it in this order is what makes
    # `capped` honest: `merge_unpaired_texts` truncates AT the budget, so a depth
    # equal to total_cap cannot by itself distinguish "trimmed" from "happened to
    # land exactly on the cap". The extra pass is a text walk, negligible next to
    # the hhfilter calls above.
    stacked = merge_unpaired_texts(
        *(per_kind_text[k] for k in KIND_ORDER), total_budget=NO_CAP
    )
    depth_uncapped = count_a3m_records(stacked)
    if total_cap is not None and depth_uncapped > total_cap:
        final = head_a3m_records(stacked, total_cap)
        info["capped"] = True
    else:
        final = stacked
        info["capped"] = False
    info["depth_before_cap"] = depth_uncapped
    info["final_depth"] = count_a3m_records(final)
    return final, info


def leveled_merge_chain(
    cid: int,
    *,
    effective_dbs: list[str],
    per_db_dir_of: dict[str, Path],
    order: list[str],
    kind_of: Callable[[str], str],
    scratch: Path,
    hhfilter_cfg: dict | None = None,
    total_cap: int | None = None,
    hhfilter_bin: str | None = None,
) -> tuple[str, dict]:
    """Merge one chain's per-DB a3ms the leveled way. Returns ``(text, info)``.

    Args:
        cid: chain index; the per-DB file is ``per_db_dir_of[db] / f"{cid}.a3m"``.
        effective_dbs: the DBs this run actually searched.
        per_db_dir_of: db_key -> directory holding that DB's per-chain a3ms.
        order: DB keys in dedup-priority order (callers pass
            ``[d for d in MERGE_ORDER if d in effective_dbs]``); a key not in
            `effective_dbs` is skipped, an effective DB with no file raises.
        kind_of: db_key -> ``"uniref"`` | ``"env"``. Engine-specific (see module
            docstring) — do NOT default it to the registry kind.
        scratch: writable dir for hhfilter's temporaries (per chain).
        hhfilter_cfg: ``{id, cov_hi, cov_lo, depth_thresh, min_keep}``; None ->
            `DEFAULT_HHFILTER_CFG`.
        total_cap: outer head-cap in records (query included), applied AFTER
            filtering; None -> uncapped.
        hhfilter_bin: explicit hhfilter path, or None to resolve from PATH.

    Returns:
        ``(merged_a3m_text, info)``. `info` carries the per-kind hhfilter stats
        (`n_in`/`n_dedup`/`cov`/`n_out`/`reverted`/`skipped`), `final_depth`, and
        `capped` (whether `total_cap` truncated the result) — all of which land in
        `method_log.merge` so a run's depth can be explained after the fact.
        Empty `order` intersection -> ``("", info)`` with `final_depth` 0.
    """
    per_db_a3m = collect_per_db_a3m(
        cid, effective_dbs=effective_dbs, per_db_dir_of=per_db_dir_of, priority=order
    )
    if not per_db_a3m:
        return "", {"kinds": {}, "final_depth": 0, "total_cap": total_cap,
                    "capped": False}
    _assert_shared_query(per_db_a3m)

    unknown = {kind_of(db) for db, _ in per_db_a3m} - set(KIND_ORDER)
    if unknown:
        raise ValueError(f"kind_of returned unknown kind(s) {sorted(unknown)}; "
                         f"expected one of {list(KIND_ORDER)}")

    texts_by_kind: dict[str, list[str]] = {k: [] for k in KIND_ORDER}
    for db, path in per_db_a3m:
        texts_by_kind[kind_of(db)].append(Path(path).read_text())
    return leveled_merge_texts(
        texts_by_kind, scratch=scratch, hhfilter_cfg=hhfilter_cfg,
        total_cap=total_cap, hhfilter_bin=hhfilter_bin,
    )
