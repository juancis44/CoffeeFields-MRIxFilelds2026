"""Bridge run_submission.py's flat output to build_submission.py's expected layout.

run_submission writes:
    <inference>/<MOD>/<TGT>/P_<MOD>_0.1T_<ID>.nii.gz          (flat)
build_submission (official) expects:
    <inference>/task2_0.1T_to_<TGT>_<MOD>/cut/pro_pretrained/epoch100/
        P_<MOD>_0.1T_<ID>.nii.gz

We create the expected tree with SYMLINKS (no copy: predictions can be large and
/home is nearly full). The official script then does the field rename + the
mandatory axial z-clip. We do NOT reimplement either -- this only reshapes dirs.

    # inspect what exists and what would be linked
    python scripts/prepare_submission_tree.py --inference_dir /data/juan/inference \
        --seg_dir /data/juan/inference_seg --dry_run

    # create the links
    python scripts/prepare_submission_tree.py --inference_dir /data/juan/inference \
        --seg_dir /data/juan/inference_seg
"""

import argparse
from pathlib import Path

MODS = ["T1W", "T2W", "T2FLAIR"]
TARGETS = ["1.5T", "3T", "5T", "7T"]
SRC = "0.1T"
METHOD, MODE, EPOCH = "cut", "pro_pretrained", "epoch100"   # build_submission defaults


def link_dir(flat: Path, nested: Path, dry: bool, warnings: list):
    """Point nested/<epoch> at the flat dir via a symlink."""
    if not flat.is_dir():
        warnings.append(f"missing flat dir: {flat}")
        return 0
    n = len(list(flat.glob("*.nii.gz")))
    if n == 0:
        warnings.append(f"no .nii.gz in {flat}")
        return 0
    if dry:
        print(f"  [dry] {nested}  ->  {flat}   ({n} files)")
        return n
    nested.parent.mkdir(parents=True, exist_ok=True)
    if nested.is_symlink() or nested.exists():
        nested.unlink()
    nested.symlink_to(flat.resolve(), target_is_directory=True)
    print(f"  {nested}  ->  {flat}   ({n} files)")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inference_dir", required=True)
    ap.add_argument("--seg_dir", default=None, help="segmentations (mirror tree); omit to skip seg")
    ap.add_argument("--task", default="task2")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    inf = Path(args.inference_dir)
    seg = Path(args.seg_dir) if args.seg_dir else None
    warnings = []

    # sanity: show the real filenames so we can confirm the seg suffix
    sample_mod, sample_tgt = MODS[0], TARGETS[0]
    print("Sample filenames actually on disk:")
    for label, base in [("pred", inf / sample_mod / sample_tgt),
                        ("seg ", (seg / sample_mod / sample_tgt) if seg else None)]:
        if base and base.is_dir():
            names = [p.name for p in sorted(base.glob("*.nii.gz"))[:3]]
            print(f"  {label}: {names if names else '(none)'}   in {base}")
        elif base:
            print(f"  {label}: MISSING dir {base}")
    print(f"\nExpected by build_submission: pred 'P_<MOD>_{SRC}_<ID>.nii.gz', "
          f"seg 'P_<MOD>_{SRC}_<ID>_seg.nii.gz'\n")

    total_pred = total_seg = 0
    for mod in MODS:
        for tgt in TARGETS:
            pair = f"{SRC}_to_{tgt}"
            base = f"{args.task}_{pair}_{mod}"
            nested_pred = inf / base / METHOD / MODE / EPOCH
            total_pred += link_dir(inf / mod / tgt, nested_pred, args.dry_run, warnings)
            if seg:
                nested_seg = seg / base / METHOD / MODE / EPOCH
                total_seg += link_dir(seg / mod / tgt, nested_seg, args.dry_run, warnings)

    print(f"\n{'='*60}\nlinked pred dirs total files: {total_pred}"
          + (f", seg: {total_seg}" if seg else ""))
    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"  - {w}")
    print(f"{'='*60}")
    print("\nNext: run the OFFICIAL packager (it does the field rename + axial z-clip):\n"
          f"  python Submission/build_submission/build_submission.py --tasks {args.task} \\\n"
          f"    --predictions-dir {inf} \\\n"
          + (f"    --predictions-seg-dir {seg} \\\n" if seg else "")
          + "    --output-dir /data/juan/submission --dry-run")


if __name__ == "__main__":
    main()
