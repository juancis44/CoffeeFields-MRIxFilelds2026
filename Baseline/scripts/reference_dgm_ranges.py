"""Build a REAL reference range for DGM structure sizes, from Validating_prospective's
own target-field volumes + their SynthSeg segmentations (target_seg/).

check_dgm_labels.py so far only tells us "present vs 0 voxels" for a GENERATED
prediction. That is a coarse signal -- it catches collapse (T2FLAIR) but says
nothing about whether a present structure is a PLAUSIBLE size.

This script segments no new data -- it just reads the real, healthy anatomy
already available locally (Validating_prospective/target_seg, real subjects
0004/0005/0008 at 1.5T etc.) and reports the real voxel-count range per
structure per field. That range is a reference, NOT ground truth for any of
your predicted subjects specifically (the subject IDs never overlap -- see the
pairing-bug finding), so use it only to sanity-check "is our predicted
structure size in a plausible ballpark", not to compute Dice/Volume against.

    python scripts/reference_dgm_ranges.py \
        --target_seg_dir /data/juan/MRIxFields2026_data/target_seg \
        --modalities T1W T2W T2FLAIR --fields 1.5T 3T 5T 7T
"""

import argparse
from pathlib import Path

import numpy as np

DGM_LABELS = {
    "L_Thalamus": 10, "R_Thalamus": 49,
    "L_Caudate": 11, "R_Caudate": 50,
    "L_Putamen": 12, "R_Putamen": 51,
    "L_Pallidum": 13, "R_Pallidum": 52,
    "L_Hippocampus": 17, "R_Hippocampus": 53,
    "L_Amygdala": 18, "R_Amygdala": 54,
    "L_Accumbens": 26, "R_Accumbens": 58,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_seg_dir", required=True,
                    help="Validating_prospective/target_seg (real target-field segmentations)")
    ap.add_argument("--modalities", nargs="+", default=["T1W", "T2W", "T2FLAIR"])
    ap.add_argument("--fields", nargs="+", default=["1.5T", "3T", "5T", "7T"])
    args = ap.parse_args()

    import nibabel as nib

    root = Path(args.target_seg_dir)

    for mod in args.modalities:
        print(f"\n{'='*90}\n{mod}  (reference: real target-field anatomy, NOT your predicted subjects)\n{'='*90}")
        for field in args.fields:
            d = root / mod / field
            segs = sorted(d.glob("*.nii.gz")) if d.is_dir() else []
            if not segs:
                print(f"  {field}: no segs in {d}")
                continue
            per_struct = {name: [] for name in DGM_LABELS}
            vox_mm3_list = []
            for s in segs:
                img = nib.load(str(s))
                v = img.get_fdata().astype(np.int32)
                vox_mm3_list.append(float(np.prod(img.header.get_zooms()[:3])))
                for name, lid in DGM_LABELS.items():
                    per_struct[name].append(int((v == lid).sum()))
            avg_vox_mm3 = float(np.mean(vox_mm3_list))
            print(f"\n  {field}  (n={len(segs)} real subjects, voxel~{avg_vox_mm3:.3f} mm^3)")
            print(f"    {'structure':>15} {'min':>8} {'median':>8} {'max':>8} |"
                  f" {'min_mm3':>9} {'med_mm3':>9} {'max_mm3':>9}")
            for name, counts in per_struct.items():
                arr = np.array(counts)
                mm3 = arr * np.array(vox_mm3_list)
                print(f"    {name:>15} {arr.min():>8} {int(np.median(arr)):>8} {arr.max():>8} |"
                      f" {mm3.min():>9.0f} {int(np.median(mm3)):>9} {mm3.max():>9.0f}")

    print(f"\n{'='*90}")
    print("Uso: compara el conteo de tu predicción (check_dgm_labels.py) contra este rango.")
    print("Muy por debajo del min real -> estructura subdimensionada / posible colapso.")
    print("Muy por encima del max real -> posible sobre-segmentacion / alucinacion de tejido.")
    print("Dentro del rango -> tamano plausible (no prueba que sea el sujeto correcto, solo que")
    print("la anatomia generada tiene una escala realista para ese campo).")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
