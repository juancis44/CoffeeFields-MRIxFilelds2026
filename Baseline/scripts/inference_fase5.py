"""Fase 5 inference — test-time ensembling (near-free metric gains).

Combines three orthogonal, single-model ensembling tricks (no extra trained
models, so it respects the Task 2 "up to 12 models" rule):

    1. Flip TTA        — average predictions over {identity, hflip, vflip}.
    2. Multi-view      — reconstruct along axial / coronal / sagittal and
                         average (uses the 3D MNI bbox to crop each view).
    3. Checkpoint avg  — average EMA weights of several checkpoints (SWA-style)
                         before inference (--checkpoints a.pth,b.pth,...).

Notes
-----
* Multi-view (`--views axial,coronal,sagittal`) helps most if the model saw all
  orientations in training. A model trained only on axial slices still usually
  gains from flip TTA; enable extra views and compare on `pro_val`. Default is
  axial only.
* Output layout matches inference_fase2.py (brain-masked, original orientation),
  so segment/submission tooling is unchanged.

    python scripts/inference_fase5.py \
        --config configs/task2/nafnet_semi/0.1T_to_3T_T1W.yaml \
        --checkpoints $OUT/task2_.../nafnet_semi/semi_last.pth \
        --input_dir $DATA_DIR/Validating_prospective/T1W/0.1T/ \
        --output_dir $INFERENCE_DIR/ --use_ema --tta --views axial
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mrixfields.env import load_env, get_inference_dir, get_device, get_preprocessed_dir
from mrixfields.data.utils import load_nifti, save_nifti
from mrixfields.data.brain_bbox import load_bbox
from mrixfields.data.histmatch import build_reference, match_histogram
from mrixfields.models.nafnet import build_nafnet


# axis -> (in-plane bbox keys) : which 3D-bbox ranges crop the 2D view
AXIS_CROP = {
    2: ("row", "col"),    # axial:    slice along dim2, in-plane (dim0,dim1)
    1: ("row", "slice"),  # coronal:  slice along dim1, in-plane (dim0,dim2)
    0: ("col", "slice"),  # sagittal: slice along dim0, in-plane (dim1,dim2)
}
VIEW_AXIS = {"axial": 2, "coronal": 1, "sagittal": 0}


def _crop2d(img, rr, cc):
    r0, r1 = rr; c0, c1 = cc
    out = np.zeros((r1 - r0, c1 - c0), np.float32)
    sr0, sr1 = max(0, r0), min(img.shape[0], r1)
    sc0, sc1 = max(0, c0), min(img.shape[1], c1)
    out[sr0 - r0:sr0 - r0 + (sr1 - sr0), sc0 - c0:sc0 - c0 + (sc1 - sc0)] = img[sr0:sr1, sc0:sc1]
    return out


def _paste2d(dst, pred, rr, cc):
    r0, r1 = rr; c0, c1 = cc
    sr0, sr1 = max(0, r0), min(dst.shape[0], r1)
    sc0, sc1 = max(0, c0), min(dst.shape[1], c1)
    dst[sr0:sr1, sc0:sc1] = pred[sr0 - r0:sr0 - r0 + (sr1 - sr0), sc0 - c0:sc0 - c0 + (sc1 - sc0)]


def _slice(vol, ax, i):
    return vol[i, :, :] if ax == 0 else (vol[:, i, :] if ax == 1 else vol[:, :, i])


@torch.no_grad()
def predict_axis(model, vol01, ax, K, bbox, device, tta):
    """Reconstruct the whole volume by 2.5D inference along `ax`."""
    rr = bbox[AXIS_CROP[ax][0]]
    cc = bbox[AXIS_CROP[ax][1]]
    n = vol01.shape[ax]
    out = np.zeros_like(vol01)
    for i in range(n):
        chans = []
        for off in range(-K, K + 1):
            j = min(max(i + off, 0), n - 1)
            chans.append(_crop2d(_slice(vol01, ax, j), rr, cc))
        stack = np.stack(chans, 0).astype(np.float32)

        variants = [stack]
        if tta:
            variants += [stack[:, ::-1, :].copy(), stack[:, :, ::-1].copy()]
        acc = None
        for k, v in enumerate(variants):
            x = torch.from_numpy(v).unsqueeze(0).to(device) * 2 - 1
            p = model(x).clamp(-1, 1).squeeze(0).squeeze(0).cpu().numpy()
            if k == 1:   # undo hflip
                p = p[::-1, :]
            elif k == 2:  # undo vflip
                p = p[:, ::-1]
            acc = p if acc is None else acc + p
        pred01 = ((acc / len(variants)) + 1) * 0.5

        dst = _slice(out, ax, i)
        _paste2d(dst, pred01, rr, cc)
    return out


def average_ema(checkpoints, device):
    """SWA-style average of EMA (or model) weights across checkpoints."""
    states = []
    for c in checkpoints:
        ck = torch.load(c, map_location=device)
        states.append(ck.get("ema", ck["model"]))
    avg = {k: sum(s[k].float() for s in states) / len(states) for k in states[0]}
    return avg


def main():
    ap = argparse.ArgumentParser(description="Fase 5 ensembled inference")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoints", required=True, help="Comma-separated .pth (averaged if >1)")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--bbox", default=None)
    ap.add_argument("--use_ema", action="store_true")
    ap.add_argument("--tta", action="store_true", help="Flip TTA")
    ap.add_argument("--views", default="axial", help="axial[,coronal[,sagittal]]")
    ap.add_argument("--match_hist", default=None,
                    help="dir with REAL target-field volumes; harmonize predictions to their "
                         "population intensity distribution (fixes the input-anchored level bias)")
    ap.add_argument("--match_hist_n", type=int, default=20,
                    help="how many reference volumes to build the distribution from")
    load_env()
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    K = cfg["data"]["n_neighbors"]
    device = torch.device((args.device or get_device()) if torch.cuda.is_available() else "cpu")
    bbox = load_bbox(args.bbox or str(Path(get_preprocessed_dir()) / "bbox.json"))
    axes = [VIEW_AXIS[v.strip()] for v in args.views.split(",") if v.strip()]

    model = build_nafnet(cfg["model"]).to(device).eval()
    ckpts = args.checkpoints.split(",")
    if len(ckpts) > 1:
        state = average_ema(ckpts, device)
        print(f"Averaged {len(ckpts)} checkpoints (SWA)")
    else:
        ck = torch.load(ckpts[0], map_location=device)
        state = ck["ema"] if (args.use_ema and "ema" in ck) else ck["model"]
    model.load_state_dict(state)

    out_dir = Path(args.output_dir or get_inference_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    in_dir = Path(args.input_dir)
    files = sorted(in_dir.rglob("*.nii.gz"))
    if not files:
        print(f"No .nii.gz in {in_dir}"); return

    hist_ref = None
    if args.match_hist:
        q_grid, ref_q, n_used, n_vox = build_reference(args.match_hist, n_volumes=args.match_hist_n)
        hist_ref = (q_grid, ref_q)
        print(f"Histogram matching ON: reference from {n_used} real volumes in "
              f"{args.match_hist} ({n_vox:,} brain voxels), "
              f"p50={np.quantile(ref_q,0.5):.3f} p95={np.quantile(ref_q,0.95):.3f}")

    for path in tqdm(files, desc=f"Inference (views={args.views}, tta={args.tta})"):
        orig = nib.load(str(path))
        orig_ornt = nib.io_orientation(orig.affine)
        data, canon_affine = load_nifti(path)
        canon_ornt = nib.io_orientation(canon_affine)

        preds = [predict_axis(model, data.astype(np.float32), ax, K, bbox, device, args.tta)
                 for ax in axes]
        pred = np.mean(preds, axis=0)

        tf = nib.orientations.ornt_transform(canon_ornt, orig_ornt)
        pred = nib.orientations.apply_orientation(pred, tf)
        src = orig.get_fdata(dtype=np.float32)
        brain = src > 1e-6
        pred = pred * brain.astype(pred.dtype)
        if hist_ref is not None:
            pred = match_histogram(pred, brain, hist_ref[0], hist_ref[1])

        save_nifti(pred, orig.affine, out_dir / path.relative_to(in_dir), header=orig.header)

    print(f"Done -> {out_dir}")


if __name__ == "__main__":
    main()
