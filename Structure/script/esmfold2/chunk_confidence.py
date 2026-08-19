"""Run the ESMFold2 confidence head one chunk of diffusion samples at a time.

Ported from CSSB-SNU/esm `scripts/chunk_confidence.py` (a fork of
github.com/Biohub/esm, MIT); see NOTICE.

Why: the confidence head, not the trunk, sets the fold's memory peak. Measured
upstream on H2343 (773 residues, num_diffusion_samples=8, A100) it is entered at
2.32 GB and peaks at 50.86 GB -- the global peak -- while the trunk peaks at
9.96 GB and a diffusion step at 6.29 GB. The cause is
``pair = _repeat_batch(z_base, S)``, which materialises an (S, L, L, d_pair)
tensor (4.55 GiB at L=773, S=8, fp32) and then runs a folding trunk over it, so
the peak grows with num_diffusion_samples until it OOMs.

That tensor genuinely differs per sample -- the per-sample predicted coordinates
enter through ``dist_bin_pairwise_embed`` -- so it cannot be shared. But every
step after the replication (trunk, row-attention pooling, pLDDT / PAE / pTM
heads) is independent across samples, so the samples need not be resident at
once: the head can run per chunk and the (small) per-sample outputs concatenated.

Only the sample axis of ``x_pred`` is split; every other input is base-batch and
the head replicates it internally for the chunk it is given.
"""

from __future__ import annotations

from typing import Callable

import torch


def install_chunked_confidence(model, chunk: int = 1) -> Callable[[], None]:
    """Patch ``model.confidence_head.forward`` to run in sample chunks.

    Returns a revert thunk; ``revert.stats`` reports how often it engaged. A
    call it cannot prove safe to split is passed through untouched:

    - the model calls the head with keywords only, so positional args or a
      missing ``x_pred`` / ``z`` mean this is not the call shape assumed here;
    - base batch must be 1, because ``_repeat_batch`` orders the flattened batch
      as ``b*S + s`` -- concatenating chunk results along dim 0 only reproduces
      that ordering for a single protein;
    - ``num_diffusion_samples <= chunk`` has nothing to split.
    """
    head = getattr(model, "confidence_head", None)
    if head is None or chunk < 1:
        return lambda: None

    orig = head.forward
    stats = {"calls": 0, "chunked": 0, "chunks": 0, "skipped": 0}

    def fwd(*args, **kw):
        stats["calls"] += 1
        if args or "x_pred" not in kw or "z" not in kw:
            stats["skipped"] += 1
            return orig(*args, **kw)

        n_samples = kw.get("num_diffusion_samples", 1)
        x_pred = kw["x_pred"]
        base = kw["z"].shape[0]
        # x_pred reaches the head either with an explicit sample axis
        # (B, S, n_atoms, 3) or already flattened to (B*S, n_atoms, 3) -- the
        # sampler currently hands over the flattened form.
        if x_pred.dim() == 4 and x_pred.shape[1] == n_samples:
            def take_samples(i: int, n: int):
                return x_pred[:, i : i + n]
        elif x_pred.dim() == 3 and x_pred.shape[0] == base * n_samples:
            def take_samples(i: int, n: int):
                return x_pred[i : i + n]
        else:
            take_samples = None  # type: ignore[assignment]

        if n_samples <= chunk or base != 1 or take_samples is None:
            stats["skipped"] += 1
            return orig(*args, **kw)

        stats["chunked"] += 1
        parts = []
        for i in range(0, n_samples, chunk):
            take = min(chunk, n_samples - i)
            stats["chunks"] += 1
            sub = dict(kw)
            sub["x_pred"] = take_samples(i, take)
            sub["num_diffusion_samples"] = take
            parts.append(orig(**sub))

        merged: dict = {}
        for key in parts[0]:
            vals = [p[key] for p in parts]
            merged[key] = (
                torch.cat(vals, dim=0) if torch.is_tensor(vals[0]) else vals[0]
            )
        return merged

    head.forward = fwd

    def revert() -> None:
        head.forward = orig

    revert.stats = stats  # type: ignore[attr-defined]
    return revert
