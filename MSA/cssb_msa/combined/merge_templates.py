"""Stage-2 template merge for the `mmseqs_hhblits_cssb` mode.

Both template engines (`cssb_template.engine_mmseqs.run_mmseqs_for_msa_dir`,
`cssb_template.engine_hmmer.run_hmmer_for_msa_dir`) emit the identical disk
contract into ``<src>/<target>_env/``:

    pdb70.m8                       # 13-col TSV; col0=qid(101+cid),
                                   #   col1="<pdb_id>_<auth>", col10=evalue
    templates_<qid>/<pdb_id>.cif   # one gunzipped cif per hit pdb

This module consolidates the two sources' ``<target>_env/`` into the combined
mode's single ``<msa_dir>/<target>_env/`` that the downstream data-yaml writer
(`colab_a3m_to_yaml.parse_m8_top_templates`) scans.

Selection per chain (qid), **mmseqs-priority**:
  1. take mmseqs rows (in the engine's e-value-ranked file order),
  2. then append hhblits rows whose ``(pdb_id, auth_chain)`` is not already
     present (a template hit is identified per-chain by pdb id + chain),
  3. stop at ``final_n`` rows for that qid.

``final_n`` is the number of templates that actually reach the model — the
downstream ``parse_m8_top_templates`` keeps only ``top_n_templates`` (default 4)
per chain by e-value. We pre-select exactly ``final_n`` mmseqs-first rows and
copy only their cifs, so the downstream e-value re-sort+cap is a no-op on the
SET: the model sees the mmseqs-prioritized templates. (mmseqs rows are NOT
intra-deduped — single-mode mmseqs behavior is preserved; only hhblits rows are
dropped on a collision with an already-selected hit.)
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# m8 column indices (shared contract; cf. `cssb_template/engine_mmseqs.py`'s
# `_COL_*` and `cssb_template/m8_writer.py`).
_COL_QUERY, _COL_TARGET, _COL_EVALUE = 0, 1, 10


def _parse_m8_by_qid(m8_path: Path) -> dict[int, list[str]]:
    """Group raw m8 rows by integer qid (col0), preserving file order.

    Missing file → empty mapping. Rows with a non-integer col0 are skipped.
    """
    by_qid: dict[int, list[str]] = {}
    m8_path = Path(m8_path)
    if not m8_path.is_file():
        return by_qid
    for line in m8_path.read_text().splitlines():
        if not line.strip():
            continue
        cols = line.split("\t")
        try:
            qid = int(cols[_COL_QUERY])
        except (ValueError, IndexError):
            continue
        by_qid.setdefault(qid, []).append(line)
    return by_qid


def _target_key(row: str) -> tuple[str, str]:
    """``(pdb_id, auth_chain)`` from a row's target col ("8bwl_A" → ("8bwl","A"))."""
    target = row.split("\t")[_COL_TARGET]
    pdb_id, _, auth = target.partition("_")
    return pdb_id, auth


def merge_template_envs_multi(
    env_sources: list[tuple[Path, str]],
    out_env: Path,
    *,
    final_n: int = 4,
) -> dict[int, dict[str, int]]:
    """Merge N ordered ``<target>_env/`` dirs into ``out_env`` (source-priority).

    Args:
        env_sources: ordered ``[(env_path, label), ...]`` — earlier source wins.
            The FIRST source is kept as-is (no intra-dedup, preserving mmseqs
            single-mode multi-alignment behavior); every later source fills only
            slots whose ``(pdb_id, auth_chain)`` is not already selected. A
            missing dir/m8 contributes nothing.
        out_env: combined output ``<target>_env/`` (cleaned of any prior
            ``templates_*`` + ``pdb70.m8`` first).
        final_n: per-chain (qid) template cap = downstream ``top_n_templates``.

    Returns:
        Per-qid stats ``{qid: {<label>: n, ..., "total": n}}``.

    Writes ``out_env/pdb70.m8`` (selected rows, priority order) + copies each
    selected row's cif into ``out_env/templates_<qid>/<pdb_id>.cif``.
    """
    if final_n < 1:
        raise ValueError(f"final_n must be >= 1, got {final_n}")
    out_env = Path(out_env)
    # (env, label, {qid: rows}, intra_dedup) — first source keeps duplicates.
    parsed = [
        (Path(env), label, _parse_m8_by_qid(Path(env) / "pdb70.m8"), i > 0)
        for i, (env, label) in enumerate(env_sources)
    ]
    qids: set[int] = set()
    for _e, _l, by_qid, _d in parsed:
        qids |= set(by_qid)
    qids_sorted = sorted(qids)

    out_env.mkdir(parents=True, exist_ok=True)
    for old in out_env.glob("templates_*"):
        shutil.rmtree(old, ignore_errors=True)
    (out_env / "pdb70.m8").unlink(missing_ok=True)

    merged_lines: list[str] = []
    stats: dict[int, dict[str, int]] = {}

    for qid in qids_sorted:
        selected: list[tuple[str, Path, str]] = []  # (row, cif_src, pdb_id)
        seen: set[tuple[str, str]] = set()
        counts = {label: 0 for _e, label, _b, _d in parsed}
        missing_cif: list[str] = []

        for src_env, label, by_qid, intra_dedup in parsed:
            if len(selected) >= final_n:
                break
            tdir_src = src_env / f"templates_{qid}"
            for row in by_qid.get(qid, []):
                if len(selected) >= final_n:
                    break
                pdb_id, auth = _target_key(row)
                if not pdb_id or not auth:
                    continue
                key = (pdb_id, auth)
                if intra_dedup and key in seen:
                    continue
                cif_src = tdir_src / f"{pdb_id}.cif"
                if not cif_src.is_file():
                    missing_cif.append(f"{label}:{pdb_id}_{auth}")
                    continue
                seen.add(key)
                selected.append((row, cif_src, pdb_id))
                counts[label] += 1

        if missing_cif and len(selected) < final_n:
            logger.warning(
                "template merge: qid=%d under-filled (%d/%d) — %d row(s) skipped "
                "for a missing cif on disk: %s",
                qid, len(selected), final_n, len(missing_cif), missing_cif,
            )

        if selected:
            tdir_out = out_env / f"templates_{qid}"
            tdir_out.mkdir(parents=True, exist_ok=True)
            for row, cif_src, pdb_id in selected:
                dst = tdir_out / f"{pdb_id}.cif"
                if not dst.exists():
                    shutil.copy2(cif_src, dst)
                merged_lines.append(row)
        counts["total"] = len(selected)
        stats[qid] = counts

    (out_env / "pdb70.m8").write_text(
        ("\n".join(merged_lines) + "\n") if merged_lines else ""
    )
    return stats


def merge_template_envs(
    mmseqs_env: Path,
    hhblits_env: Path,
    out_env: Path,
    *,
    final_n: int = 4,
) -> dict[int, dict[str, int]]:
    """Two-source (mmseqs-priority) wrapper over `merge_template_envs_multi`."""
    return merge_template_envs_multi(
        [(mmseqs_env, "mmseqs"), (hhblits_env, "hhblits")], out_env, final_n=final_n
    )
