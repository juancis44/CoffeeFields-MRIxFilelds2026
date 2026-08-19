"""Brain bounding-box utilities.

All volumes live in a common MNI grid (364 x 436 x 364, 0.5 mm), so a *single*
in-plane bounding box computed once applies to every subject / field / modality.
Cropping the background:
    - removes ~40-60% of empty voxels  -> less memory, faster training,
    - stops the large black border from *inflating* SSIM / deflating nRMSE
      (a model can score well on background it never had to reconstruct).

The bbox is stored as JSON:
    {"axis": 2, "row": [r0, r1], "col": [c0, c1], "slice": [s0, s1]}
where row/col index the in-plane (H, W) axes of an axial slice and slice is the
non-empty range along the stacking axis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
#  Compute
# --------------------------------------------------------------------------- #
def foreground_bbox_2d(mask: np.ndarray, margin: int = 4) -> Tuple[int, int, int, int]:
    """Return (r0, r1, c0, c1) tight box around True voxels, padded by `margin`."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return 0, mask.shape[0], 0, mask.shape[1]
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    r0 = max(0, r0 - margin); c0 = max(0, c0 - margin)
    r1 = min(mask.shape[0], r1 + 1 + margin)
    c1 = min(mask.shape[1], c1 + 1 + margin)
    return int(r0), int(r1), int(c0), int(c1)


def accumulate_bbox_from_slices(
    slice_iter,
    thresh: float = 0.02,
    margin: int = 4,
) -> Dict:
    """Fold a stream of 2D slices into a global in-plane bbox + non-empty slice range.

    Args:
        slice_iter: iterable of (slice_idx:int, image:np.ndarray[H,W] in [0,1]).
        thresh: intensity above which a voxel is foreground.
        margin: padding in voxels around the union box.
    """
    r0g = c0g = 10**9
    r1g = c1g = -1
    smin, smax = 10**9, -1
    for s_idx, img in slice_iter:
        m = img > thresh
        if not m.any():
            continue
        r0, r1, c0, c1 = foreground_bbox_2d(m, margin=0)
        r0g, c0g = min(r0g, r0), min(c0g, c0)
        r1g, c1g = max(r1g, r1), max(c1g, c1)
        smin, smax = min(smin, s_idx), max(smax, s_idx)

    if r1g < 0:
        raise ValueError("No foreground found — check `thresh`.")

    # apply margin now, clamped later by the caller against real dims
    r0g = max(0, r0g - margin); c0g = max(0, c0g - margin)
    r1g = r1g + margin; c1g = c1g + margin
    return {
        "row": [int(r0g), int(r1g)],
        "col": [int(c0g), int(c1g)],
        "slice": [int(smin), int(smax + 1)],
        "thresh": thresh,
        "margin": margin,
    }


# --------------------------------------------------------------------------- #
#  I/O + apply
# --------------------------------------------------------------------------- #
def save_bbox(bbox: Dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(bbox, f, indent=2)


def load_bbox(path: str | Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def snap_to_multiple(bbox: Dict, multiple: int = 8) -> Dict:
    """Expand row/col ranges so their size is divisible by `multiple`.

    Convnets/UNets need dims divisible by 2^depth; do it once here instead of
    padding every batch.
    """
    out = dict(bbox)
    for key in ("row", "col"):
        a, b = bbox[key]
        size = b - a
        pad = (-size) % multiple
        out[key] = [a, b + pad]
    return out


def apply_bbox(img: np.ndarray, bbox: Dict) -> np.ndarray:
    """Crop a 2D (H,W) or 2.5D (C,H,W) array to the in-plane bbox.

    Uses safe clamping + edge padding so the output size is exactly
    (row[1]-row[0], col[1]-col[0]) even if the box exceeds the array.
    """
    r0, r1 = bbox["row"]; c0, c1 = bbox["col"]
    H_out, W_out = r1 - r0, c1 - c0
    if img.ndim == 2:
        return _crop_pad_2d(img, r0, r1, c0, c1, H_out, W_out)
    if img.ndim == 3:  # (C, H, W)
        return np.stack([_crop_pad_2d(img[c], r0, r1, c0, c1, H_out, W_out)
                         for c in range(img.shape[0])], axis=0)
    raise ValueError(f"Expected 2D or 3D array, got shape {img.shape}")


def _crop_pad_2d(img, r0, r1, c0, c1, H_out, W_out):
    H, W = img.shape
    out = np.zeros((H_out, W_out), dtype=img.dtype)
    sr0, sr1 = max(0, r0), min(H, r1)
    sc0, sc1 = max(0, c0), min(W, c1)
    dr0, dc0 = sr0 - r0, sc0 - c0
    out[dr0:dr0 + (sr1 - sr0), dc0:dc0 + (sc1 - sc0)] = img[sr0:sr1, sc0:sc1]
    return out
