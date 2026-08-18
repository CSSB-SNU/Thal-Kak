# Thal-Kak
# Copyright 2026 CSSB, Seoul National University
#
# Licensed under the Apache License, Version 2.0 (see LICENSE).
#
# ---------------------------------------------------------------------------
# Third-party attribution (see also NOTICE)
#
# plot_msa_coverage() and plot_msa_coverage_multimer() in this file are a
# Python port of the monomer and multimer branches of ColabFold's
# plot_msa_v2() (colabfold/plot.py,
# https://github.com/sokrypton/ColabFold):
#
#   MIT License
#
#   Copyright (c) 2021 Sergey Ovchinnikov
#
#   Permission is hereby granted, free of charge, to any person obtaining a
#   copy of this software and associated documentation files (the "Software"),
#   to deal in the Software without restriction, including without limitation
#   the rights to use, copy, modify, merge, publish, distribute, sublicense,
#   and/or sell copies of the Software, and to permit persons to whom the
#   Software is furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#   DEALINGS IN THE SOFTWARE.
#
# MODIFIED by CSSB Thal-Kak (2026): the two functions consume ASCII-int8 a3m
# matrices (gap = ord('-') = 45) built from this pipeline's per-chain a3m
# files, rather than an AlphaFold `feature_dict` msa array (gap = 21); they
# use matplotlib's object-oriented API instead of pyplot global state; they
# add a `sort_lines` option, raise on empty input instead of producing an
# empty figure, and title each figure with its N/L/chain counts. The
# reproduced ColabFold conventions — rows sorted by identity to query,
# gap cells masked to NaN and scaled by row identity, the rainbow_r
# [0,1] heatmap with the "Sequence identity to query" colorbar, the black
# per-position coverage curve, and (multimer) grouping rows by per-chain
# gap pattern with black chain/group border lines — are unchanged.
#
# Everything else in this file is original to this work: _expand_to_multimer()
# (which builds the multimer matrix from per-chain a3m files, a step ColabFold
# does not have because it reads a prepared feature_dict), plot_neff_curve(),
# PlotResult, _render_per_db_plots(), _detect_target_and_layout() and
# plot_local_msa().
# ---------------------------------------------------------------------------

"""Coverage heatmap + per-position Neff curve plotting for local_msa.

matplotlib is lazy-imported inside each plot function so that callers
that only need numpy-side `neff.py` don't pull matplotlib. Backend is
forced to "Agg" for SLURM/headless use.

Output style mirrors ColabFold's `colabfold.plot.plot_msa_v2` (rainbow_r
heatmap, identity colorbar, black coverage curve overlay) but consumes
ASCII-int8 a3m matrices (gap = ord('-') = 45) instead of feature_dict.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from MSA.local_msa.common.dedup import A3M_GAP_BYTE
from MSA.local_msa.plot.neff import compute_neff, load_a3m_as_int_matrix

logger = logging.getLogger(__name__)


def plot_msa_coverage(
    arr: np.ndarray,
    query_seq: np.ndarray,
    out_png: Path | str,
    *,
    dpi: int = 100,
    sort_lines: bool = True,
) -> None:
    """ColabFold-style MSA coverage heatmap.

    Rows = sequences (sorted by identity to query, ascending), columns =
    positions. Cell color = per-row identity to query (rainbow_r), gap
    cells masked. A black per-position non-gap-count curve is overlaid.

    arr:        (N, L) int8 — a3m dedup-key bytes (uppercase + '-').
    query_seq:  (L,) int8 — first row's bytes; identity is computed
                relative to this.
    out_png:    write target.
    sort_lines: True (default) sorts rows by identity ascending; False
                shows input order reversed (newest at top).
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    N, L = arr.shape
    if N == 0 or L == 0:
        raise ValueError(f"empty matrix (N={N}, L={L})")

    nogap = arr != A3M_GAP_BYTE                       # (N, L) bool
    qid = arr == query_seq[None, :]                   # (N, L) bool
    query_nogap = query_seq != A3M_GAP_BYTE           # (L,) bool

    n_query_cols = max(int(query_nogap.sum()), 1)
    seqid = (qid & query_nogap[None, :]).sum(-1) / n_query_cols

    non_gaps = nogap.astype(float)
    non_gaps[non_gaps == 0] = np.nan
    order = np.argsort(seqid) if sort_lines else np.arange(N)[::-1]
    lines = non_gaps[order] * seqid[order, None]

    coverage_curve = (~np.isnan(lines)).sum(0)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)
    ax.set_title(f"MSA coverage (N={N}, L={L})")
    im = ax.imshow(
        lines,
        interpolation="nearest",
        aspect="auto",
        cmap="rainbow_r",
        vmin=0,
        vmax=1,
        origin="lower",
        extent=(0, L, 0, N),
    )
    ax.plot(coverage_curve, color="black", linewidth=1.0)
    ax.set_xlim(0, L)
    ax.set_ylim(0, N)
    fig.colorbar(im, ax=ax, label="Sequence identity to query")
    ax.set_xlabel("Positions")
    ax.set_ylabel("Sequences")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def _expand_to_multimer(
    paired_per_chain: list[np.ndarray] | None,
    unpaired_per_chain: list[np.ndarray],
    cardinality: list[int],
) -> tuple[np.ndarray, list[int]]:
    """Build a multimer-expanded `(N_total, sum(L_i × C_i))` int8 matrix.

    Multimer-style convention (used by AlphaFold2-multimer, AF3, Boltz2,
    Chai1, Protenix during multimer feature construction): homo-oligomer
    copies share MSA hits.

      * Paired records: each chain block is replicated `cardinality_i`
        times in one row so a single paired record fills all chain copy
        slots — same as ColabFold `batch.pair_sequences`.
      * Unpaired record of unique chain `u`: ONE row, body placed
        identically in all `cardinality_u` copy slots of chain u, every
        other chain's slots gap-padded.

    For A1B1 (cardinality = [1, 1]) this reduces to one row per chain
    slot (no copy duplication). For A2B2 the row count is
    `N_paired + N_unp_0 + N_unp_1` (not 2× the unpaired counts).

    Note: this is a *visualization* helper. It does not modify any a3m
    file. The downstream models read the on-disk split a3m files and do
    their own multimer feature processing.

    paired_per_chain:   list of (N_paired, L_i) int8 arrays, all sharing
                        the same `N_paired`. None or empty for monomer /
                        homomer where pairing doesn't apply.
    unpaired_per_chain: list of (N_unp_i, L_i) int8 arrays, one per
                        unique chain.
    cardinality:        copy count per unique chain.

    Returns:
        expanded: (N_total, total_L) int8.
        Ls:       per-slot chain lengths (len = sum(cardinality)).
    """
    n_unique = len(unpaired_per_chain)
    if len(cardinality) != n_unique:
        raise ValueError(
            f"cardinality len ({len(cardinality)}) != n_unique ({n_unique})"
        )

    chain_layout = [
        (u, c)
        for u in range(n_unique)
        for c in range(cardinality[u])
    ]
    Ls = [unpaired_per_chain[u].shape[1] for u, _ in chain_layout]
    total_L = sum(Ls)
    offsets = np.cumsum([0] + Ls)

    blocks: list[np.ndarray] = []

    has_paired = (
        paired_per_chain is not None
        and len(paired_per_chain) > 0
        and paired_per_chain[0].shape[0] > 0
    )
    if has_paired:
        N_paired = paired_per_chain[0].shape[0]
        if not all(p.shape[0] == N_paired for p in paired_per_chain):
            raise ValueError("paired_per_chain rows must match across chains")
        if any(p.shape[1] != unpaired_per_chain[u].shape[1]
               for u, p in enumerate(paired_per_chain)):
            raise ValueError("paired/unpaired chain lengths disagree")
        paired_block = np.full((N_paired, total_L), A3M_GAP_BYTE, dtype=np.int8)
        for s, (u, _) in enumerate(chain_layout):
            paired_block[:, offsets[s]:offsets[s + 1]] = paired_per_chain[u]
        blocks.append(paired_block)

    # Unpaired: one block per unique chain. Body replicated into all of
    # that chain's copy slots within each row (multimer-style sharing).
    slot_indices_for_chain = [
        [s for s, (uu, _) in enumerate(chain_layout) if uu == u]
        for u in range(n_unique)
    ]
    for u in range(n_unique):
        unp = unpaired_per_chain[u]
        if unp.shape[0] == 0:
            continue
        block = np.full((unp.shape[0], total_L), A3M_GAP_BYTE, dtype=np.int8)
        for s in slot_indices_for_chain[u]:
            block[:, offsets[s]:offsets[s + 1]] = unp
        blocks.append(block)

    if not blocks:
        return np.zeros((0, total_L), dtype=np.int8), Ls
    return np.concatenate(blocks, axis=0), Ls


def plot_msa_coverage_multimer(
    paired_per_chain: list[np.ndarray] | None,
    unpaired_per_chain: list[np.ndarray],
    cardinality: list[int],
    out_png: Path | str,
    *,
    dpi: int = 100,
    sort_lines: bool = True,
) -> None:
    """ColabFold-style multimer-expanded coverage heatmap.

    Builds the multimer matrix (paired chain blocks × cardinality + per-
    slot unpaired blocks) via `_expand_to_multimer`, then groups rows by
    chain coverage pattern (per-chain non-gap mask) and sorts each group
    by identity to query. Visual style matches `plot_msa_v2`:
      * vertical black lines at chain borders,
      * horizontal black lines between gap-pattern groups,
      * black coverage curve overlay,
      * rainbow_r heatmap with identity colorbar.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    expanded, Ls = _expand_to_multimer(
        paired_per_chain, unpaired_per_chain, cardinality
    )
    N, total_L = expanded.shape
    if N == 0 or total_L == 0:
        raise ValueError(f"empty multimer matrix (N={N}, L={total_L})")

    Ln = np.cumsum([0] + Ls)
    n_chains = len(Ls)

    query = expanded[0]
    nogap = expanded != A3M_GAP_BYTE
    qid = expanded == query[None, :]

    # Per-chain gap pattern: True if any non-gap residue in that chain's
    # column block.
    gapid = np.stack(
        [nogap[:, Ln[c]:Ln[c + 1]].max(-1) for c in range(n_chains)],
        axis=-1,
    )

    lines_groups: list[np.ndarray] = []
    group_sizes: list[int] = []
    for pattern in np.unique(gapid, axis=0):
        if not pattern.any():
            continue
        idx = np.where((gapid == pattern).all(axis=-1))[0]
        if len(idx) == 0:
            continue
        sub_qid = qid[idx]
        per_chain_id = np.stack(
            [sub_qid[:, Ln[c]:Ln[c + 1]].mean(-1) for c in range(n_chains)],
            axis=-1,
        )
        seqid = per_chain_id.sum(-1) / max(int(pattern.sum()), 1)
        sub_nogap = nogap[idx].astype(float)
        sub_nogap[sub_nogap == 0] = np.nan
        order = np.argsort(seqid) if sort_lines else np.arange(len(idx))[::-1]
        lines_groups.append(sub_nogap[order] * seqid[order, None])
        group_sizes.append(len(idx))

    if not lines_groups:
        raise ValueError("no rows with any non-gap chain coverage")

    Nn = np.cumsum([0] + group_sizes)
    lines = np.concatenate(lines_groups, axis=0)
    coverage_curve = (~np.isnan(lines)).sum(0)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)
    ax.set_title(f"MSA coverage (N={lines.shape[0]}, L={total_L}, chains={n_chains})")
    im = ax.imshow(
        lines,
        interpolation="nearest",
        aspect="auto",
        cmap="rainbow_r",
        vmin=0,
        vmax=1,
        origin="lower",
        extent=(0, total_L, 0, lines.shape[0]),
    )
    for x in Ln[1:-1]:
        ax.plot([x, x], [0, lines.shape[0]], color="black", linewidth=1.0)
    for y in Nn[1:-1]:
        ax.plot([0, total_L], [y, y], color="black", linewidth=1.0)
    ax.plot(coverage_curve, color="black", linewidth=1.0)
    ax.set_xlim(0, total_L)
    ax.set_ylim(0, lines.shape[0])
    fig.colorbar(im, ax=ax, label="Sequence identity to query")
    ax.set_xlabel("Positions")
    ax.set_ylabel("Sequences")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_neff_curve(
    neff_per_position: np.ndarray,
    out_png: Path | str,
    *,
    neff_scalar: float | None = None,
    per_db: dict[str, np.ndarray] | None = None,
    dpi: int = 100,
) -> None:
    """Per-position Neff line plot.

    Black thick line for the merged curve; optional thin colored lines
    for per-DB breakdown overlaid.

    neff_per_position: (L,) array (e.g. from compute_neff(...).per_position).
    neff_scalar:       scalar Neff for the legend, optional.
    per_db:            dict of `{db_label: (L,) array}` for overlay.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    arr = np.asarray(neff_per_position)
    if arr.ndim != 1:
        raise ValueError(f"neff_per_position must be 1-D, got shape {arr.shape}")
    L = arr.shape[0]
    if L == 0:
        raise ValueError("empty neff_per_position")

    fig, ax = plt.subplots(figsize=(8, 4), dpi=dpi)
    label = (
        f"merged (Neff={neff_scalar:.1f})"
        if neff_scalar is not None
        else "merged"
    )
    ax.plot(arr, color="black", linewidth=1.5, label=label)

    if per_db:
        cmap = plt.get_cmap("tab10")
        for i, (name, curve) in enumerate(sorted(per_db.items())):
            ax.plot(
                np.asarray(curve),
                color=cmap(i % 10),
                linewidth=0.8,
                alpha=0.7,
                label=name,
            )

    ax.set_xlim(0, L)
    ax.set_xlabel("Positions")
    ax.set_ylabel("Neff (effective # of sequences)")
    ax.set_title("Per-position Neff")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


@dataclass(frozen=True)
class PlotResult:
    plots_dir: Path
    coverage_pngs: list[Path] = field(default_factory=list)
    neff_pngs: list[Path] = field(default_factory=list)
    multimer_png: Path | None = None
    per_db_coverage_pngs: list[Path] = field(default_factory=list)
    per_db_neff_pngs: list[Path] = field(default_factory=list)
    summary_json: Path | None = None
    skipped: bool = False


def _render_per_db_plots(
    msa_dir: Path,
    plots_dir: Path,
    u: int,
    chain_str: str,
    *,
    identity_threshold: float,
    subsample_cap: int | None,
    subsample_seed: int,
    dpi: int,
) -> tuple[
    dict[str, np.ndarray] | None,
    dict[str, dict] | None,
    dict[str, list[Path]],
]:
    """Render coverage + Neff plots for each per-DB raw a3m of chain `u`.

    Walks `_workdir/raw/<db>/<u>.a3m` (created by `build_a3m` with
    `keep_intermediate=True`) and writes
    `plots_dir/per_db/<db>/{chain_str}_coverage.png` +
    `{chain_str}_neff.png` for every DB whose raw a3m has more than the
    query row. The `pair/` subdir (paired multimer output, rows aligned
    across chains) is intentionally skipped — it needs a different
    plotter.

    Returns:
        per_db_curves:  `{db_key: per_position_neff}` for the merged-Neff
                        overlay (or `None` if nothing rendered).
        per_db_summary: `{db_key: {n_total, n_used, L, neff_scalar,
                        subsampled}}` for `neff_summary.json` (or `None`).
        pngs:           `{"coverage": [...], "neff": [...]}` PNG paths
                        for the caller to attach to `PlotResult`.
    """
    raw_root = msa_dir / "_workdir" / "raw"
    pngs: dict[str, list[Path]] = {"coverage": [], "neff": []}
    if not raw_root.is_dir():
        logger.warning(
            "_workdir/raw missing under %s — per-DB plots skipped", msa_dir
        )
        return None, None, pngs

    per_db_curves: dict[str, np.ndarray] = {}
    per_db_summary: dict[str, dict] = {}
    per_db_root = plots_dir / "per_db"

    for db_dir in sorted(raw_root.iterdir()):
        if not db_dir.is_dir():
            continue
        db_key = db_dir.name
        if db_key == "pair":
            # Paired multimer output — chain-aligned rows, different shape.
            continue
        a3m_path = db_dir / f"{u}.a3m"
        if not a3m_path.exists():
            continue
        try:
            arr, query = load_a3m_as_int_matrix(a3m_path)
        except Exception as exc:
            logger.warning(
                "per-DB load failed (%s, chain %s): %s",
                db_key, chain_str, exc,
            )
            continue
        if arr.shape[0] <= 1:
            # Only the query row — nothing meaningful to plot.
            continue

        out_dir = per_db_root / db_key
        out_dir.mkdir(parents=True, exist_ok=True)

        cov_png = out_dir / f"{chain_str}_coverage.png"
        try:
            q_bytes = np.frombuffer(query.encode("ascii"), dtype=np.int8)
            plot_msa_coverage(arr, q_bytes, cov_png, dpi=dpi)
            pngs["coverage"].append(cov_png)
        except Exception as exc:
            logger.warning(
                "per-DB coverage plot failed (%s, chain %s): %s",
                db_key, chain_str, exc,
            )

        try:
            res = compute_neff(
                arr,
                identity_threshold=identity_threshold,
                subsample_cap=subsample_cap,
                seed=subsample_seed,
            )
        except Exception as exc:
            logger.warning(
                "per-DB Neff failed (%s, chain %s): %s",
                db_key, chain_str, exc,
            )
            continue

        per_db_curves[db_key] = res.per_position
        per_db_summary[db_key] = {
            "n_total": int(res.n_total),
            "n_used": int(res.n_used),
            "L": int(arr.shape[1]),
            "neff_scalar": float(res.scalar),
            "subsampled": bool(res.subsampled),
        }

        neff_png = out_dir / f"{chain_str}_neff.png"
        try:
            plot_neff_curve(
                res.per_position,
                neff_png,
                neff_scalar=res.scalar,
                dpi=dpi,
            )
            pngs["neff"].append(neff_png)
        except Exception as exc:
            logger.warning(
                "per-DB Neff plot failed (%s, chain %s): %s",
                db_key, chain_str, exc,
            )

    return (per_db_curves or None), (per_db_summary or None), pngs


def _detect_target_and_layout(
    msa_dir: Path,
    target_hint: str | None,
) -> tuple[str, list[int], list[int], list[str]]:
    """Detect (target, unique chain lengths, cardinality, chain-letter
    strings per entity) from the master a3m header `#L1,L2,...\\tC1,C2,...`.
    """
    if target_hint:
        target = target_hint
    else:
        candidates = [
            p for p in msa_dir.glob("*.yaml")
            if p.name != "method_log.yaml"
        ]
        target = next(
            (c.stem for c in candidates if (msa_dir / f"{c.stem}.a3m").exists()),
            None,
        )
        if target is None:
            raise FileNotFoundError(
                f"no <target>.yaml + <target>.a3m pair found in {msa_dir}"
            )

    master_a3m = msa_dir / f"{target}.a3m"
    if not master_a3m.exists():
        raise FileNotFoundError(f"master a3m missing: {master_a3m}")

    with open(master_a3m) as f:
        first_line = f.readline().strip()
    if not first_line.startswith("#"):
        raise ValueError(
            f"master a3m missing colab header line in {master_a3m!s}: {first_line!r}"
        )
    parts = first_line[1:].split("\t")
    if len(parts) != 2:
        raise ValueError(f"invalid colab a3m header: {first_line!r}")
    unique_lengths = [int(x) for x in parts[0].split(",")]
    cardinality = [int(x) for x in parts[1].split(",")]
    if len(unique_lengths) != len(cardinality):
        raise ValueError(f"length/cardinality count mismatch: {first_line!r}")

    # Chain-letter strings per entity match `colab_a3m_to_yaml`'s interleaved
    # convention: chains are assigned copy-major across entities, not
    # partitioned per entity. e.g.
    #   A2B2 → entity0=[A,C]→"a_c", entity1=[B,D]→"b_d"
    #   A2B1 → entity0=[A,C]→"a_c", entity1=[B]→"b"
    #   A1B1 → entity0=[A]→"a",     entity1=[B]→"b"
    #   A2   → entity0=[A,B]→"a_b"  (single-entity homomer)
    from MSA.script.colab_msa_template_search.colab_a3m_to_yaml import (
        get_chain_names,
    )

    all_letters = get_chain_names(sum(cardinality))
    entity_chains: list[list[str]] = [[] for _ in cardinality]
    current_idx = 0
    max_copies = max(cardinality, default=0)
    for r in range(max_copies):
        for i, n in enumerate(cardinality):
            if r < n:
                entity_chains[i].append(all_letters[current_idx])
                current_idx += 1
    chain_letters_per_entity: list[str] = [
        "_".join(c).lower() for c in entity_chains
    ]

    return target, unique_lengths, cardinality, chain_letters_per_entity


def plot_local_msa(
    msa_dir: Path | str,
    *,
    target: str | None = None,
    primary: Literal["unpaired_split", "merged"] = "unpaired_split",
    identity_threshold: float = 0.62,
    subsample_cap: int | None = 10_000,
    subsample_seed: int = 0,
    do_per_db: bool = True,
    out_subdir: str = "plots",
    overwrite: bool = False,
    dpi: int = 100,
) -> PlotResult:
    """Build per-chain coverage + Neff plots (and a multimer-expanded
    coverage plot for complexes) from a `build_a3m` output directory.

    Detects target name and chain layout from the master a3m header
    `#L1,L2,...\\tC1,C2,...`. For each unique entity, writes
    `{chain_str}_coverage.png` + `{chain_str}_neff.png`. For a complex
    (n_unique > 1 or any cardinality > 1), additionally writes
    `multimer_coverage.png`. A single `neff_summary.json` records per-
    chain Neff scalars and metadata.

    When `do_per_db=True` (default), additionally renders coverage +
    Neff plots for each per-DB raw a3m under
    `_workdir/raw/<db>/<u>.a3m` into `plots/per_db/<db>/`, and overlays
    each DB's Neff curve as a thin colored line on the merged
    `{chain_str}_neff.png` plot. The `pair/` subdir is skipped (paired
    multimer rows are chain-aligned and need a different visualization).

    msa_dir:            `build_a3m` output dir (= `out_a3m.parent`).
    target:             optional override; default = stem of the
                        `<x>.yaml` whose `<x>.a3m` exists in the dir.
    primary:            `"unpaired_split"` (default; reads
                        `{target}_unpaired_msa_chains_<x>.a3m` — the
                        same file inference consumes) or `"merged"`
                        (reads `_workdir/merged/<cid>.a3m` for debug).
                        Multimer plot is only emitted in
                        `"unpaired_split"` mode.
    identity_threshold: τ for Neff clustering (HHsuite default 0.62).
    subsample_cap:      compute_neff subsample cap; None disables.
    subsample_seed:     RNG seed for subsample reproducibility.
    do_per_db:          when True (default), render per-DB coverage +
                        Neff plots from `_workdir/raw/<db>/<u>.a3m` and
                        overlay each DB on the merged Neff plot. Set
                        False to skip — only `_workdir/raw/` users see
                        a difference; primary plots are unchanged.
    out_subdir:         plots dir name under msa_dir. Default `"plots"`.
    overwrite:          when False (default), skip if `neff_summary.json`
                        exists. When True, regenerate all artifacts.
    dpi:                figure dpi (default 100 → 800×500 px coverage
                        plot, 800×400 px Neff curve).

    Returns a `PlotResult` with paths and a `skipped` flag.
    """
    msa_dir = Path(msa_dir).resolve()
    plots_dir = msa_dir / out_subdir
    summary_path = plots_dir / "neff_summary.json"

    if summary_path.exists() and not overwrite:
        return PlotResult(
            plots_dir=plots_dir,
            summary_json=summary_path,
            skipped=True,
        )

    target, unique_lengths, cardinality, chain_letters = (
        _detect_target_and_layout(msa_dir, target)
    )
    n_unique = len(unique_lengths)
    is_complex = n_unique > 1 or any(c > 1 for c in cardinality)

    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load per-chain a3m matrices once (reused for per-chain + multimer plots).
    unpaired_per_chain: list[np.ndarray] = []
    paired_per_chain: list[np.ndarray] = []
    queries: list[str] = []
    for u, chain_str in enumerate(chain_letters):
        if primary == "unpaired_split":
            unp_path = msa_dir / f"{target}_unpaired_msa_chains_{chain_str}.a3m"
        else:
            unp_path = msa_dir / "_workdir" / "merged" / f"{u}.a3m"
        if not unp_path.exists():
            raise FileNotFoundError(f"unpaired a3m missing: {unp_path}")
        arr, query = load_a3m_as_int_matrix(unp_path)
        unpaired_per_chain.append(arr)
        queries.append(query)

        if is_complex and n_unique > 1 and primary == "unpaired_split":
            paired_path = msa_dir / f"{target}_paired_msa_chains_{chain_str}.a3m"
            if paired_path.exists():
                p_arr, _ = load_a3m_as_int_matrix(paired_path)
            else:
                p_arr = np.zeros((0, unique_lengths[u]), dtype=np.int8)
            paired_per_chain.append(p_arr)

    # Per-chain plots + Neff.
    coverage_pngs: list[Path] = []
    neff_pngs: list[Path] = []
    per_db_coverage_pngs: list[Path] = []
    per_db_neff_pngs: list[Path] = []
    summary_chains: dict[str, dict] = {}
    t0 = time.time()

    for u, chain_str in enumerate(chain_letters):
        arr = unpaired_per_chain[u]
        query = queries[u]
        if arr.shape[0] == 0:
            logger.warning("empty a3m for chain %s — skipping", chain_str)
            continue
        cov_png = plots_dir / f"{chain_str}_coverage.png"
        q_bytes = np.frombuffer(query.encode("ascii"), dtype=np.int8)
        plot_msa_coverage(arr, q_bytes, cov_png, dpi=dpi)
        coverage_pngs.append(cov_png)

        res = compute_neff(
            arr,
            identity_threshold=identity_threshold,
            subsample_cap=subsample_cap,
            seed=subsample_seed,
        )

        # Per-DB plots + curves (also feed the merged-Neff overlay).
        per_db_curves: dict[str, np.ndarray] | None = None
        per_db_summary: dict[str, dict] | None = None
        if do_per_db:
            per_db_curves, per_db_summary, per_db_pngs = _render_per_db_plots(
                msa_dir,
                plots_dir,
                u,
                chain_str,
                identity_threshold=identity_threshold,
                subsample_cap=subsample_cap,
                subsample_seed=subsample_seed,
                dpi=dpi,
            )
            per_db_coverage_pngs.extend(per_db_pngs["coverage"])
            per_db_neff_pngs.extend(per_db_pngs["neff"])

        neff_png = plots_dir / f"{chain_str}_neff.png"
        plot_neff_curve(
            res.per_position,
            neff_png,
            neff_scalar=res.scalar,
            per_db=per_db_curves,
            dpi=dpi,
        )
        neff_pngs.append(neff_png)

        chain_summary: dict = {
            "chain_letters": chain_str.split("_"),
            "unique_entity_index": u,
            "cardinality": int(cardinality[u]),
            "n_total": int(res.n_total),
            "n_used": int(res.n_used),
            "L": int(arr.shape[1]),
            "neff_scalar": float(res.scalar),
            "subsampled": bool(res.subsampled),
        }
        if per_db_summary:
            chain_summary["per_db"] = per_db_summary
        summary_chains[chain_str] = chain_summary

    # Multimer-expanded plot (skip in merged mode and for true monomers).
    multimer_png: Path | None = None
    if is_complex and primary == "unpaired_split":
        try:
            multimer_target = plots_dir / "multimer_coverage.png"
            paired = (
                paired_per_chain
                if (paired_per_chain and any(p.shape[0] > 0 for p in paired_per_chain))
                else None
            )
            plot_msa_coverage_multimer(
                paired,
                unpaired_per_chain,
                list(cardinality),
                multimer_target,
                dpi=dpi,
            )
            multimer_png = multimer_target
        except Exception as exc:
            logger.warning("multimer plot failed: %s", exc)

    wall = time.time() - t0

    summary = {
        "schema_version": 2,
        "target": target,
        "msa_dir": str(msa_dir),
        "primary_source": primary,
        "identity_threshold": identity_threshold,
        "subsample": {"cap": subsample_cap, "seed": subsample_seed},
        "do_per_db": bool(do_per_db),
        "n_unique": n_unique,
        "cardinality": list(cardinality),
        "is_complex": is_complex,
        "chains": summary_chains,
        "multimer_plot": str(multimer_png.relative_to(msa_dir)) if multimer_png else None,
        "wall_seconds": round(wall, 2),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    return PlotResult(
        plots_dir=plots_dir,
        coverage_pngs=coverage_pngs,
        neff_pngs=neff_pngs,
        multimer_png=multimer_png,
        per_db_coverage_pngs=per_db_coverage_pngs,
        per_db_neff_pngs=per_db_neff_pngs,
        summary_json=summary_path,
        skipped=False,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Render per-chain + multimer MSA coverage and Neff plots "
                    "for a build_a3m output directory."
    )
    parser.add_argument("msa_dir", help="build_a3m output directory")
    parser.add_argument("--target", default=None,
                        help="override target name (default: detect from <X>.yaml)")
    parser.add_argument("--primary", default="unpaired_split",
                        choices=["unpaired_split", "merged"])
    parser.add_argument("--identity-threshold", type=float, default=0.62)
    parser.add_argument("--subsample-cap", type=int, default=10_000,
                        help="compute_neff subsample cap (0 disables)")
    parser.add_argument("--subsample-seed", type=int, default=0)
    parser.add_argument(
        "--no-per-db",
        action="store_true",
        help="skip per-DB raw a3m coverage/Neff plots and the merged-Neff overlay.",
    )
    parser.add_argument("--out-subdir", default="plots")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dpi", type=int, default=100)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = plot_local_msa(
        args.msa_dir,
        target=args.target,
        primary=args.primary,
        identity_threshold=args.identity_threshold,
        subsample_cap=args.subsample_cap if args.subsample_cap > 0 else None,
        subsample_seed=args.subsample_seed,
        do_per_db=not args.no_per_db,
        out_subdir=args.out_subdir,
        overwrite=args.overwrite,
        dpi=args.dpi,
    )
    if result.skipped:
        print(f"skipped (existing summary): {result.summary_json}")
    else:
        print(f"plots dir: {result.plots_dir}")
        for p in result.coverage_pngs:
            print(f"  coverage: {p.name}")
        for p in result.neff_pngs:
            print(f"  neff:     {p.name}")
        if result.multimer_png:
            print(f"  multimer: {result.multimer_png.name}")
        if result.per_db_coverage_pngs or result.per_db_neff_pngs:
            print(
                f"  per-DB:   {len(result.per_db_coverage_pngs)} coverage + "
                f"{len(result.per_db_neff_pngs)} neff under per_db/"
            )
        print(f"summary:   {result.summary_json}")
