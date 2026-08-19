"""Intensity harmonization: match a prediction to the target field's canonical
intensity distribution.

Why: a model trained on SYNTHETIC degradation learns a mapping anchored to the
*input's* absolute level (we measured pred ~= 0.77 x input, in lockstep across
subjects) instead of producing the target field's canonical level. That biases
nRMSE, which is an absolute-error metric.

Fix: build a reference quantile function from many REAL target-field volumes
(abundant in Training_retrospective, and independent of the unconfirmed subject
pairing), then quantile-map each prediction's brain voxels onto it. This is
standard intensity harmonization, not per-subject fitting: the reference is a
single population-level distribution reused for every prediction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def build_reference(volume_dir, n_volumes: int = 20, n_quantiles: int = 256,
                    brain_thresh: float = 0.01, max_vox_per_vol: int = 200_000,
                    seed: int = 0):
    """Population quantile function of brain intensities from real target volumes.

    Returns (q_grid, ref_quantiles) suitable for `match_histogram`.
    """
    import nibabel as nib
    files = sorted(Path(volume_dir).glob("*.nii.gz"))
    if not files:
        raise FileNotFoundError(f"No .nii.gz in {volume_dir}")
    files = files[:n_volumes] if n_volumes > 0 else files

    rng = np.random.default_rng(seed)
    pool = []
    for f in files:
        v = nib.load(str(f)).get_fdata(dtype=np.float32)
        b = v[v > brain_thresh]
        if b.size == 0:
            continue
        if b.size > max_vox_per_vol:                      # subsample to bound memory
            b = rng.choice(b, size=max_vox_per_vol, replace=False)
        pool.append(b)
    if not pool:
        raise ValueError(f"No brain voxels found under {volume_dir}")
    pool = np.concatenate(pool)

    q_grid = np.linspace(0.0, 1.0, n_quantiles)
    ref_quantiles = np.quantile(pool, q_grid).astype(np.float32)
    return q_grid, ref_quantiles, len(files), pool.size


def _strictly_increasing(x, eps=1e-7):
    """np.interp needs an increasing xp; quantiles can have ties."""
    x = np.maximum.accumulate(x)
    return x + np.arange(x.size, dtype=x.dtype) * eps


def match_histogram(pred, mask, q_grid, ref_quantiles):
    """Quantile-map `pred[mask]` onto the reference distribution.

    Background (outside mask) is left untouched (stays 0).
    """
    out = pred.copy()
    src = pred[mask]
    if src.size == 0:
        return out
    src_q = _strictly_increasing(np.quantile(src, q_grid).astype(np.float32))
    out[mask] = np.interp(src, src_q, ref_quantiles).astype(pred.dtype)
    return out
