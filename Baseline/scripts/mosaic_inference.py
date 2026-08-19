"""Visual QC mosaic: rows = contrast (T1W / T2W / T2FLAIR), columns = subject.

One figure per target field (or a stacked one over several). Each cell shows a
central brain slice of the prediction, so you can eyeball across contrasts and
subjects at once. Optionally prepend the 0.1T input and/or overlay the SynthSeg
segmentation (useful for debugging the T2FLAIR Dice/Volume = 0.25 problem: if
the seg overlay is empty or garbage, that explains the score, not the image).

Files are matched by SUBJECT ID (the trailing 4 digits), not by full filename,
because the filename embeds the modality (P_T2W_0.1T_0001 vs P_T1W_0.1T_0001).

    python scripts/mosaic_inference.py \
        --inference_dir /data/juan/inference \
        --input_root /data/juan/MRIxFields2026_data/Validating_prospective \
        --target_field 3T --out /home/juanc/challenge/mosaic_3T.png

    python scripts/mosaic_inference.py --inference_dir /data/juan/inference \
        --input_root /data/juan/MRIxFields2026_data/Validating_prospective \
        --fields 1.5T 3T 5T 7T --with_input \
        --seg_dir /data/juan/inference_seg2 --out /home/juanc/challenge/mosaic_all.png
"""

import argparse
import re
from pathlib import Path

import numpy as np

MODS = ["T1W", "T2W", "T2FLAIR"]


def load_nifti(path):
    import nibabel as nib
    img = nib.as_closest_canonical(nib.load(str(path)))
    return img.get_fdata(dtype=np.float32)


def central_slice(vol, thresh=0.05):
    z = vol.shape[2]
    content = [(vol[:, :, i] > thresh).sum() for i in range(z)]
    k = int(np.argmax(content)) if any(content) else z // 2
    s = vol[:, :, k]
    lo, hi = np.percentile(s[s > 0], [1, 99]) if (s > 0).any() else (0, 1)
    return np.clip((s - lo) / (hi - lo + 1e-8), 0, 1), k


def subject_id(name):
    m = re.search(r"(\d{4})(?:_seg)?\.nii\.gz$", name)
    return m.group(1) if m else name


def find_by_id(d: Path, sid: str, seg=False):
    """First nii.gz in d whose trailing ID matches sid (seg toggles the _seg suffix)."""
    if not d.is_dir():
        return None
    for p in sorted(d.glob("*.nii.gz")):
        if p.name.endswith("_seg.nii.gz") != seg:
            continue
        if subject_id(p.name) == sid:
            return p
    return None


def discover_ids(inf, fields, max_subjects):
    for f in fields:
        for m in MODS:
            d = inf / m / f
            if d.is_dir():
                ids = sorted({subject_id(p.name) for p in d.glob("*.nii.gz")
                              if not p.name.endswith("_seg.nii.gz")})
                if ids:
                    return ids[:max_subjects]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inference_dir", required=True)
    ap.add_argument("--input_root", default=None, help="Validating_prospective root (for 0.1T input)")
    ap.add_argument("--seg_dir", default=None, help="segmentation tree (nested layout) for overlay")
    ap.add_argument("--fields", nargs="+", default=None)
    ap.add_argument("--target_field", default="3T")
    ap.add_argument("--with_input", action="store_true")
    ap.add_argument("--max_subjects", type=int, default=3)
    ap.add_argument("--out", default="mosaic.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    inf = Path(args.inference_dir)
    fields = args.fields or [args.target_field]
    seg = Path(args.seg_dir) if args.seg_dir else None
    inroot = Path(args.input_root) if args.input_root else None

    ids = discover_ids(inf, fields, args.max_subjects)
    if not ids:
        print(f"No predictions under {inf}/<MOD>/<field> for fields {fields}.")
        return

    n_block_rows = len(MODS)
    n_rows = len(fields) * n_block_rows
    n_cols = len(ids) * (2 if args.with_input else 1)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.6 * n_rows), squeeze=False)

    for fi, field in enumerate(fields):
        for mi, mod in enumerate(MODS):
            r = fi * n_block_rows + mi
            pred_dir = inf / mod / field
            for si, sid in enumerate(ids):
                col = si * (2 if args.with_input else 1)
                # input cell
                if args.with_input:
                    ax = axes[r][col]
                    ip = find_by_id(inroot / mod / "0.1T", sid) if inroot else None
                    if ip and ip.exists():
                        img, _ = central_slice(load_nifti(ip))
                        ax.imshow(img.T, cmap="gray", origin="lower")
                    else:
                        ax.text(0.5, 0.5, "no input", ha="center", va="center", fontsize=7)
                    ax.set_xticks([]); ax.set_yticks([])
                    if r == 0:
                        ax.set_title(f"{sid}\ninput 0.1T", fontsize=7)
                    col += 1
                # prediction cell
                ax = axes[r][col]
                pp = find_by_id(pred_dir, sid)
                if pp and pp.exists():
                    img, k = central_slice(load_nifti(pp))
                    ax.imshow(img.T, cmap="gray", origin="lower")
                    if seg is not None:
                        segd = (seg / f"task2_0.1T_to_{field}_{mod}" / "cut" /
                                "pro_pretrained" / "epoch100")
                        sp = find_by_id(segd, sid, seg=True)
                        if sp and sp.exists():
                            sv = load_nifti(sp)
                            sl = sv[:, :, min(k, sv.shape[2] - 1)]
                            ax.imshow(np.ma.masked_where(sl == 0, sl).T, cmap="autumn",
                                      origin="lower", alpha=0.35)
                else:
                    ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
                if (not args.with_input and col == 0) or (args.with_input and col == 1):
                    ax.set_ylabel(f"{field}\n{mod}", fontsize=8)
                if r == 0 and not args.with_input:
                    ax.set_title(sid, fontsize=8)

    ttl = "predictions" + (" + 0.1T input" if args.with_input else "") + (" + seg overlay" if seg else "")
    fig.suptitle(f"Task 2 inference mosaic  ({ttl})  --  rows: field/contrast, cols: subject", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"Saved {args.out}  ({n_rows}x{n_cols} cells, fields={fields}, ids={ids})")


if __name__ == "__main__":
    main()
