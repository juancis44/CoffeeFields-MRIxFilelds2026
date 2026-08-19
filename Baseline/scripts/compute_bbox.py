"""Compute a global brain bounding box from preprocessed npz slices.

All volumes share the MNI grid, so one bbox (computed from a robust subset)
applies to every subject/field/modality. We compute it from the *target*
(high-field) slices, which have the clearest brain outline.

Usage
-----
    python scripts/compute_bbox.py \
        --preprocessed_dir $PREPROCESSED_DIR \
        --split pro_train --modality T1W --field 1.5T \
        --out $PREPROCESSED_DIR/bbox.json --snap 8

Then point the 2.5D dataset at the saved bbox.json.
"""

import argparse
import os
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mrixfields.data.brain_bbox import (
    accumulate_bbox_from_slices, snap_to_multiple, save_bbox,
)


def _slice_stream(base: Path, split: str, modality: str, field: str, max_subjects: int):
    d = base / split / modality / field
    files = sorted(d.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No npz under {d}")
    if max_subjects > 0:
        # keep files from the first `max_subjects` distinct subjects
        seen, kept = set(), []
        for f in files:
            subj = "_".join(f.stem.split("_")[-2:-1])  # subject id token
            if subj not in seen and len(seen) >= max_subjects:
                continue
            seen.add(subj); kept.append(f)
        files = kept
    for f in files:
        # slice index is the trailing sNNN token
        s_idx = int(f.stem.split("_")[-1][1:])
        img = np.load(f)["image"]
        yield s_idx, img


def main():
    ap = argparse.ArgumentParser(description="Compute global brain bbox from npz slices")
    ap.add_argument("--preprocessed_dir", required=True)
    ap.add_argument("--split", default="pro_train")
    ap.add_argument("--modality", default="T1W")
    ap.add_argument("--field", default="1.5T", help="Field to derive the box from (use a target field)")
    ap.add_argument("--thresh", type=float, default=0.02)
    ap.add_argument("--margin", type=int, default=4)
    ap.add_argument("--snap", type=int, default=8, help="Snap H/W to a multiple of this (0=off)")
    ap.add_argument("--max_subjects", type=int, default=0, help="0 = use all")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = Path(args.preprocessed_dir)

    # The defaults may not exist (splits get pruned / re-extracted). Fall back to
    # any populated split/field for this modality instead of dying: the bbox is
    # global (shared MNI grid), so any high-field source is equivalent.
    if not list((base / args.split / args.modality / args.field).glob("*.npz")):
        cand = None
        for sp in (args.split, "retro_train", "pro_train", "pro_val"):
            for fl in (args.field, "3T", "1.5T", "5T", "7T", "0.1T"):
                d = base / sp / args.modality / fl
                if d.is_dir() and any(d.glob("*.npz")):
                    cand = (sp, fl); break
            if cand: break
        if cand is None:
            raise FileNotFoundError(
                f"No npz found for modality {args.modality} under {base}. "
                f"Run 'preprocess.py extract-slices' first.")
        print(f"[bbox] {args.split}/{args.modality}/{args.field} is empty; "
              f"using {cand[0]}/{args.modality}/{cand[1]} instead", flush=True)
        args.split, args.field = cand

    stream = _slice_stream(base, args.split, args.modality, args.field, args.max_subjects)
    bbox = accumulate_bbox_from_slices(stream, thresh=args.thresh, margin=args.margin)
    if args.snap > 0:
        bbox = snap_to_multiple(bbox, args.snap)

    out_path = args.out
    if os.path.isdir(out_path):
        out_path = os.path.join(out_path, "bbox.json")
    save_bbox(bbox, out_path)
    H = bbox["row"][1] - bbox["row"][0]
    W = bbox["col"][1] - bbox["col"][0]
    ns = bbox["slice"][1] - bbox["slice"][0]
    print(f"bbox saved -> {out_path}")
    print(f"  in-plane crop: {H} x {W}  (row {bbox['row']}, col {bbox['col']})")
    print(f"  non-empty slices: {ns}  (range {bbox['slice']})")


if __name__ == "__main__":
    main()
