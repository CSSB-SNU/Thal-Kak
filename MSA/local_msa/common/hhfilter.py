"""Staged hhfilter (RoseTTAFold-style coverage + seq-id filter) on a3m TEXT.

`common/leveled_merge.py` calls this per (chain x DB kind) whenever
`merge.mode: leveled`, which is the default for all three local modes
(`mmseqs_local`, `hhblits_local`, `mmseqs_hhblits_local`). The binary comes from
`hhsuite` in `environment.yml`.

`hhfilter` (HH-suite) filters an a3m by MAX pairwise identity (`-id`, redundancy
removal), MIN coverage with the query (`-cov`), and — unset here — diversity
(`-diff`, default 0 = off). It operates directly on an a3m and keeps the seed
(record 0 = query) at the top.

`staged_hhfilter` applies the RoseTTAFold cov schedule with a min-keep guard:

  1. ``-id 90 -cov 75``  → if depth >= ``depth_thresh`` (2000) keep it.
  2. else ``-id 90 -cov 50`` → if depth >= ``min_keep`` (100) keep it.
  3. else (cov 50 too aggressive on a shallow group) → return the UNFILTERED
     input (dedup'd union), i.e. skip hhfilter.

Applied per group (uniref / env), per chain — never on an assembled complex a3m.
The caller stacks the filtered groups (uniref on top) and applies the
`caps.total_max` safety cap AFTER filtering. Groups with <= 1 record
(query-only/empty) skip hhfilter (needs >= 2 seqs). The `hhfilter` binary is
resolved from the active env's PATH (like `mmseqs/runner.resolve_mmseqs`).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def resolve_hhfilter(explicit: str | Path | None = None) -> str:
    """Resolve the hhfilter binary: an `explicit` path if given, else `hhfilter`
    from the active env's PATH. Fails loud if missing — no hardcoded fallback."""
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"hhfilter binary {path} not found")
        return str(path)
    found = shutil.which("hhfilter")
    if found is None:
        raise FileNotFoundError(
            "hhfilter not found on PATH; activate an env that provides HH-suite, "
            "or pass an explicit `hhfilter`"
        )
    return found


def count_seqs(a3m_text: str) -> int:
    """Number of a3m records (lines starting with '>')."""
    return sum(1 for ln in a3m_text.splitlines() if ln.startswith(">"))


def _run_hhfilter(in_a3m: Path, out_a3m: Path, *, id_cut: int, cov: int, hhfilter: str) -> str:
    cmd = [hhfilter, "-i", str(in_a3m), "-o", str(out_a3m),
           "-id", str(id_cut), "-cov", str(cov)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not out_a3m.is_file():
        raise RuntimeError(
            f"hhfilter failed (id={id_cut} cov={cov}) rc={r.returncode}: "
            f"{(r.stderr or r.stdout)[-800:]}"
        )
    return out_a3m.read_text()


def staged_hhfilter(
    a3m_text: str,
    workdir: Path | str,
    *,
    hhfilter: str | Path | None = None,
    id_cut: int = 90,
    cov_hi: int = 75,
    cov_lo: int = 50,
    depth_thresh: int = 2000,
    min_keep: int = 100,
    tag: str = "grp",
) -> tuple[str, dict]:
    """Return ``(filtered_text, info)`` for one per-chain a3m group.

    ``hhfilter`` is an explicit binary path, or ``None`` to resolve from PATH.
    ``info`` = ``{n_in, cov, n_out, n_cov_hi?, n_cov_lo?, skipped, reverted?}``.
    ``cov`` is the coverage% used (``cov_hi``/``cov_lo``), ``None`` when skipped
    (<=1 seq), or ``"none(union)"`` when reverted to the unfiltered union.
    """
    n_in = count_seqs(a3m_text)
    if n_in <= 1:
        return a3m_text, {"n_in": n_in, "cov": None, "n_out": n_in, "skipped": True}
    hhfilter = resolve_hhfilter(hhfilter)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    in_a3m = workdir / f"{tag}_in.a3m"
    in_a3m.write_text(a3m_text if a3m_text.endswith("\n") else a3m_text + "\n")

    t_hi = _run_hhfilter(in_a3m, workdir / f"{tag}_cov{cov_hi}.a3m",
                         id_cut=id_cut, cov=cov_hi, hhfilter=hhfilter)
    n_hi = count_seqs(t_hi)
    if n_hi >= depth_thresh:
        return t_hi, {"n_in": n_in, "cov": cov_hi, "n_out": n_hi, "skipped": False}

    t_lo = _run_hhfilter(in_a3m, workdir / f"{tag}_cov{cov_lo}.a3m",
                         id_cut=id_cut, cov=cov_lo, hhfilter=hhfilter)
    n_lo = count_seqs(t_lo)
    if n_lo >= min_keep:
        return t_lo, {"n_in": n_in, "cov": cov_lo, "n_out": n_lo,
                      "n_cov_hi": n_hi, "skipped": False}

    # cov_lo leaves < min_keep seqs -> don't filter; keep the unfiltered union.
    return a3m_text, {"n_in": n_in, "cov": "none(union)", "n_out": n_in,
                      "n_cov_hi": n_hi, "n_cov_lo": n_lo, "skipped": False,
                      "reverted": True}
