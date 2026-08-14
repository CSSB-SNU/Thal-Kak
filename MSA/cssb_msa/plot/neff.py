"""Neff (effective number of sequences) computation for cssb_msa.

HHsuite/AF-style sequence-level Neff:

    Neff = Σ_i 1/n_i
    n_i  = |{ j : seqid(i, j) >= τ }|     (j includes i, so n_i ≥ 1)

Identity is computed on a3m dedup-keys (uppercase + '-', length = query
L), the key form built by `common/dedup.a3m_dedup_key`:

    seqid(a, b) = matches / aligned_columns
    matches         = positions where a[i] == b[i] != gap
    aligned_columns = positions where at least one of a[i], b[i] != gap

Default threshold τ = 0.62 (HHsuite hhfilter -id 62, AF default).

Numpy-only. Its one consumer is `cssb_msa/plot/plotting.py`.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from MSA.cssb_msa.common.dedup import (
    A3M_GAP_BYTE,
    a3m_dedup_key,
    iter_a3m_records,
)


@dataclass(frozen=True, eq=False)
class NeffResult:
    scalar: float
    per_position: np.ndarray   # (L,) float64
    n_total: int               # rows in the input matrix
    n_used: int                # rows actually used (<= n_total when subsampled)
    identity_threshold: float
    subsampled: bool


def load_a3m_as_int_matrix(path: Path | str) -> tuple[np.ndarray, str]:
    """Read an a3m file and return ((N, L) int8 matrix, query_str).

    Lowercase insertion columns are stripped (= AF3 dedup-key form). Each
    row in the matrix is the query-aligned subsequence as ASCII bytes
    (uppercase letters + '-'); gap byte = ord('-') = 45.

    Returns ``(np.zeros((0, 0), int8), "")`` for an empty file. Raises
    ``ValueError`` if any record's dedup-key length differs from the
    first record's — that indicates real a3m corruption, not silent
    skip.
    """
    text = Path(path).read_text()
    records = list(iter_a3m_records(text))
    if not records:
        return np.zeros((0, 0), dtype=np.int8), ""
    keys = [a3m_dedup_key(body) for _, body in records]
    L = len(keys[0])
    bad = next((i for i, k in enumerate(keys) if len(k) != L), None)
    if bad is not None:
        raise ValueError(
            f"length mismatch in {path}: row {bad} key len {len(keys[bad])} "
            f"!= query len {L}"
        )
    flat = "".join(keys).encode("ascii")
    arr = np.frombuffer(flat, dtype=np.int8).reshape(len(keys), L)
    return arr, keys[0]


def compute_neff(
    arr: np.ndarray,
    *,
    identity_threshold: float = 0.62,
    subsample_cap: int | None = 10_000,
    seed: int = 0,
    chunk: int = 256,
) -> NeffResult:
    """Sequence-level Neff (HHsuite/AF style) on a (N, L) int8 matrix.

    Per-row cluster size:
        n_i = |{ j : seqid(i, j) >= identity_threshold }|     (j incl. i)
    Scalar:
        Neff = Σ_i 1/n_i
    Per-position effective coverage:
        neff_pos[r] = Σ_{i: arr[i, r] != gap} 1/n_i

    Memory: pairwise comparison is row-block chunked. Each `(chunk, N, L)`
    bool array costs chunk*N*L bytes — ~1 GB at chunk=256, N=10k, L=400 —
    and up to three coexist per iteration (`aligned`, `matches`, and the
    equality temporary), so budget ~3× that. Drop chunk if you see OOM.

    Subsample: if `subsample_cap` is not None and N > cap, uniformly
    sample (cap-1) non-query rows (without replacement) and always keep
    row 0 (query). Sets `subsampled=True`. `seed` controls the RNG for
    reproducibility.
    """
    N, L = arr.shape
    if N == 0 or L == 0:
        return NeffResult(
            scalar=0.0,
            per_position=np.zeros(L, dtype=np.float64),
            n_total=N,
            n_used=N,
            identity_threshold=identity_threshold,
            subsampled=False,
        )

    n_total = N
    subsampled = False
    if subsample_cap is not None and N > subsample_cap:
        rng = np.random.default_rng(seed)
        idx = np.concatenate([
            np.array([0]),
            rng.choice(N - 1, size=subsample_cap - 1, replace=False) + 1,
        ])
        idx.sort()
        arr = arr[idx]
        N = arr.shape[0]
        subsampled = True

    nogap = arr != A3M_GAP_BYTE                # (N, L) bool, computed once
    n_i = np.zeros(N, dtype=np.float64)
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        block = arr[start:end]                                     # (c, L)
        nogap_b = nogap[start:end]                                 # (c, L)
        aligned = nogap_b[:, None, :] | nogap[None, :, :]          # (c, N, L)
        matches = (block[:, None, :] == arr[None, :, :]) & nogap[None, :, :]
        n_aligned = aligned.sum(-1)                                # (c, N)
        n_matches = matches.sum(-1)                                # (c, N)
        ids = np.where(
            n_aligned > 0,
            n_matches / np.maximum(n_aligned, 1),
            0.0,
        )
        n_i[start:end] = (ids >= identity_threshold).sum(-1).astype(np.float64)

    # n_i[i] >= 1 for any row with at least one non-gap residue (self-id
    # is 1.0). Floor protects against the all-gap edge case.
    inv = 1.0 / np.maximum(n_i, 1.0)
    neff_scalar = float(inv.sum())
    neff_per_position = (inv[:, None] * nogap.astype(np.float64)).sum(0)

    return NeffResult(
        scalar=neff_scalar,
        per_position=neff_per_position,
        n_total=n_total,
        n_used=N,
        identity_threshold=identity_threshold,
        subsampled=subsampled,
    )
