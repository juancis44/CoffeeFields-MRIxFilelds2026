#!/usr/bin/env python3
"""MRIxFields2026 Task 2 — Docker inference entrypoint.

Reads /input/manifest.json, runs 2.5D NAFNet inference with EMA weights
and flip TTA for each sample, writes full-volume predictions to /output/.

Container I/O contract (participant_guide.md):
    /input/manifest.json           manifest with sample list
    /input/{MOD}/{pair}/src/...    source NIfTI volumes (read-only)
    /output/{MOD}/{pair}/pred/...  prediction NIfTI volumes (writable)

Output spec:
    Shape:     (364, 436, 364)  — full 3D volume, no z-clipping
    Dtype:     float32
    Range:     [0, 1]
    Format:    .nii.gz
    No segmentation files.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

from nafnet import build_nafnet

# --------------------------------------------------------------------------- #
#  Model configuration (fixed across all 12 Task 2 checkpoints)
# --------------------------------------------------------------------------- #
MODEL_CFG = {
    "input_nc": 5,
    "output_nc": 1,
    "width": 32,
    "enc_blk_nums": [2, 2, 4],
    "middle_blk_num": 6,
    "dec_blk_nums": [2, 2, 2],
    "global_residual": True,
    "drop": 0.0,
}
N_NEIGHBORS = 2  # K=2 -> 5 input channels (center + 2 on each side)

APP_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = APP_DIR / "weights"
BBOX_PATH = APP_DIR / "bbox.json"

INPUT_ROOT = Path("/input")
OUTPUT_ROOT = Path("/output")

# Axial axis = 2; row/col are dims 0/1 of each axial slice
AXIS_CROP = {2: ("row", "col"), 1: ("row", "slice"), 0: ("col", "slice")}


# --------------------------------------------------------------------------- #
#  NIfTI I/O
# --------------------------------------------------------------------------- #
def load_nifti(path):
    """Load NIfTI, reorient to RAS+ canonical. Returns (data, affine)."""
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    return img.get_fdata(dtype=np.float32), img.affine


def save_nifti(data, affine, path, header=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data.astype(np.float32), affine, header=header)
    nib.save(img, str(path))


# --------------------------------------------------------------------------- #
#  Bounding box
# --------------------------------------------------------------------------- #
def load_bbox():
    """Load brain bounding box; fall back to full volume if missing."""
    if BBOX_PATH.exists():
        with open(BBOX_PATH) as f:
            bbox = json.load(f)
        print(f"  bbox.json loaded: row={bbox['row']}, col={bbox['col']}")
        return bbox
    print("  bbox.json not found — using full-volume fallback (slower)")
    return {"row": [0, 364], "col": [0, 436], "slice": [0, 364]}


# --------------------------------------------------------------------------- #
#  2.5D slice helpers
# --------------------------------------------------------------------------- #
def _crop2d(img, rr, cc):
    r0, r1 = rr
    c0, c1 = cc
    out = np.zeros((r1 - r0, c1 - c0), np.float32)
    sr0 = max(0, r0)
    sr1 = min(img.shape[0], r1)
    sc0 = max(0, c0)
    sc1 = min(img.shape[1], c1)
    out[sr0 - r0 : sr0 - r0 + (sr1 - sr0),
        sc0 - c0 : sc0 - c0 + (sc1 - sc0)] = img[sr0:sr1, sc0:sc1]
    return out


def _paste2d(dst, pred, rr, cc):
    r0, r1 = rr
    c0, c1 = cc
    sr0 = max(0, r0)
    sr1 = min(dst.shape[0], r1)
    sc0 = max(0, c0)
    sc1 = min(dst.shape[1], c1)
    dst[sr0:sr1, sc0:sc1] = pred[
        sr0 - r0 : sr0 - r0 + (sr1 - sr0),
        sc0 - c0 : sc0 - c0 + (sc1 - sc0),
    ]


def _get_slice(vol, ax, i):
    if ax == 0:
        return vol[i, :, :]
    elif ax == 1:
        return vol[:, i, :]
    return vol[:, :, i]


# --------------------------------------------------------------------------- #
#  2.5D axial inference with flip TTA
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_volume(model, vol, bbox, device):
    """Run 2.5D axial inference with flip TTA (identity + hflip + vflip)."""
    K = N_NEIGHBORS
    ax = 2
    rr = bbox[AXIS_CROP[ax][0]]
    cc = bbox[AXIS_CROP[ax][1]]
    n = vol.shape[ax]
    out = np.zeros_like(vol)

    for i in range(n):
        chans = []
        for off in range(-K, K + 1):
            j = min(max(i + off, 0), n - 1)
            chans.append(_crop2d(_get_slice(vol, ax, j), rr, cc))
        stack = np.stack(chans, 0).astype(np.float32)

        v_identity = stack
        v_hflip = stack[:, ::-1, :].copy()
        v_vflip = stack[:, :, ::-1].copy()
        batch = np.stack([v_identity, v_hflip, v_vflip], axis=0)

        x = torch.from_numpy(batch).to(device) * 2 - 1
        preds = model(x).clamp(-1, 1).squeeze(1).cpu().numpy()

        p0 = preds[0]
        p1 = preds[1, ::-1, :]
        p2 = preds[2, :, ::-1]
        pred01 = ((p0 + p1 + p2) / 3.0 + 1.0) * 0.5

        dst = _get_slice(out, ax, i)
        _paste2d(dst, pred01, rr, cc)

    return out


# --------------------------------------------------------------------------- #
#  Weight file resolution
# --------------------------------------------------------------------------- #
def get_weight_path(source_field, target_field, modality):
    name = f"task2_{source_field}_to_{target_field}_{modality}_da_semi_last.pth"
    path = WEIGHTS_DIR / name
    if path.exists():
        return path
    available = sorted(p.name for p in WEIGHTS_DIR.glob("*.pth"))
    raise FileNotFoundError(
        f"Weight file not found: {name}\nAvailable: {available}"
    )


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    manifest_path = INPUT_ROOT / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    samples = manifest["samples"]
    task = manifest.get("task", "unknown")
    print(f"Task: {task}  |  Samples: {len(samples)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    bbox = load_bbox()

    groups = defaultdict(list)
    for s in samples:
        key = (s["source_field"], s["target_field"], s["modality"])
        groups[key].append(s)

    print(f"Model groups: {len(groups)}")

    total_done = 0
    for (src, tgt, mod), group_samples in groups.items():
        print(f"\n--- {mod}: {src} -> {tgt} ({len(group_samples)} samples) ---")

        weight_path = get_weight_path(src, tgt, mod)
        print(f"  Weights: {weight_path.name}")

        model = build_nafnet(MODEL_CFG).to(device).eval()
        ck = torch.load(str(weight_path), map_location=device, weights_only=False)
        state = ck.get("ema", ck.get("model", ck))
        model.load_state_dict(state)

        for sample in tqdm(group_samples, desc=f"  {mod} {src}->{tgt}"):
            input_path = INPUT_ROOT / sample["input"]
            output_path = OUTPUT_ROOT / sample["output"]

            orig = nib.load(str(input_path))
            orig_ornt = nib.io_orientation(orig.affine)

            data, canon_affine = load_nifti(input_path)
            canon_ornt = nib.io_orientation(canon_affine)

            pred = predict_volume(model, data, bbox, device)

            tf = nib.orientations.ornt_transform(canon_ornt, orig_ornt)
            pred = nib.orientations.apply_orientation(pred, tf)

            src_data = orig.get_fdata(dtype=np.float32)
            brain = src_data > 1e-6
            pred = pred * brain.astype(pred.dtype)

            pred = np.clip(pred, 0.0, 1.0)

            save_nifti(pred, orig.affine, output_path, header=orig.header)
            total_done += 1

        del model, ck, state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nDone: {total_done}/{len(samples)} predictions written to {OUTPUT_ROOT}")

    missing = [s["output"] for s in samples if not (OUTPUT_ROOT / s["output"]).exists()]
    if missing:
        print(f"ERROR: {len(missing)} missing outputs:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        sys.exit(1)

    print("All outputs verified.")


if __name__ == "__main__":
    main()
