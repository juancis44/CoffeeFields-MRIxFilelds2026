"""Check whether the SCORED structures are actually present in our segmentations.

Dice/Volume in this challenge are NOT whole-brain metrics. evaluate.py's
compute_dice/compute_volume_consistency only look at 14 deep-gray-matter (DGM)
labels: thalamus, caudate, putamen, pallidum, hippocampus, amygdala, accumbens
(left+right). A segmentation can cover the whole brain (what the mosaic shows)
and still score near the scorer's floor if these small subcortical structures
are missing, mis-labelled, or malformed.

This prints, per modality/field, how many voxels SynthSeg assigned to each DGM
label -- so we can see whether the structures exist at all before assuming the
low Dice/Volume is an anatomical-accuracy problem rather than a "label missing
entirely" problem.

    python scripts/check_dgm_labels.py \
        --seg_dir /data/juan/inference_seg2 \
        --modalities T1W T2W T2FLAIR --fields 3T 7T
"""

import argparse
from pathlib import Path

import numpy as np

# same table as Evaluation/evaluate.py::DGM_LABELS
DGM_LABELS = {
    "L_Thalamus": 10, "R_Thalamus": 49,
    "L_Caudate": 11, "R_Caudate": 50,
    "L_Putamen": 12, "R_Putamen": 51,
    "L_Pallidum": 13, "R_Pallidum": 52,
    "L_Hippocampus": 17, "R_Hippocampus": 53,
    "L_Amygdala": 18, "R_Amygdala": 54,
    "L_Accumbens": 26, "R_Accumbens": 58,
}

METHOD, MODE, EPOCH = "cut", "pro_pretrained", "epoch100"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg_dir", required=True)
    ap.add_argument("--modalities", nargs="+", default=["T1W", "T2W", "T2FLAIR"])
    ap.add_argument("--fields", nargs="+", default=["1.5T", "3T", "5T", "7T"])
    ap.add_argument("--n_subjects", type=int, default=3)
    ap.add_argument("--mm3", action="store_true",
                    help="also print physical volume (mm^3) using the NIfTI voxel size -- "
                         "use this when comparing against reference_dgm_ranges.py, since raw "
                         "voxel counts are NOT comparable across different segmentation grids "
                         "(e.g. this pipeline's 0.5mm-resampled output vs target_seg's native "
                         "~1mm SynthSeg grid is an ~8x voxel-count difference for the SAME "
                         "physical volume).")
    args = ap.parse_args()

    import nibabel as nib

    seg_root = Path(args.seg_dir)
    all_labels = sorted(set(DGM_LABELS.values()))

    for mod in args.modalities:
        print(f"\n{'='*90}\n{mod}\n{'='*90}")
        header = f"{'field':>6} {'subject':>10} | " + " ".join(f"{n[:8]:>8}" for n in DGM_LABELS)
        print(header)
        for field in args.fields:
            d = seg_root / f"task2_0.1T_to_{field}_{mod}" / METHOD / MODE / EPOCH
            segs = sorted(d.glob("*_seg.nii.gz")) if d.is_dir() else []
            if not segs:
                print(f"{field:>6}  (no seg files in {d})")
                continue
            n_zero_labels_total = 0
            for s in segs[:args.n_subjects]:
                img = nib.load(str(s))
                v = img.get_fdata().astype(np.int32)
                vox_mm3 = float(np.prod(img.header.get_zooms()[:3]))
                present_bg = int((v > 0).sum())
                counts = []
                n_zero = 0
                for name, lid in DGM_LABELS.items():
                    c = int((v == lid).sum())
                    counts.append(c)
                    if c == 0:
                        n_zero += 1
                n_zero_labels_total += n_zero
                subj = s.name.split("_")[-1].replace("_seg.nii.gz", "")
                flag = "  <-- ALL DGM LABELS MISSING" if n_zero == len(DGM_LABELS) else (
                    f"  <-- {n_zero}/{len(DGM_LABELS)} missing" if n_zero > 0 else "")
                print(f"{field:>6} {subj:>10} | " + " ".join(f"{c:>8}" for c in counts) + flag)
                if args.mm3:
                    mm3_row = " ".join(f"{c * vox_mm3:>8.0f}" for c in counts)
                    print(f"{'':>6} {'(mm^3)':>10} | " + mm3_row
                          + f"   [voxel={vox_mm3:.3f} mm^3]")
            print(f"{'':>6} {'':>10}   (whole-seg nonzero voxels for scale, last subj: {present_bg})")

    print(f"\n{'='*90}")
    print("Read-out: if DGM voxel counts are 0 or near-0 across the board for a")
    print("modality, SynthSeg is not resolving those structures in that contrast --")
    print("the whole-brain 'rescue' we saw does not fix this, since it targets")
    print("overall coverage, not these specific small nuclei. If counts are present")
    print("but small/noisy, the issue is segmentation ACCURACY on tiny structures,")
    print("which usually needs a less aggressive denoise (less blur = keeps more of")
    print("the (noisy but present) structure boundary) or a sharper prediction image")
    print("upstream (the lowfreq retrain), not more smoothing.")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
