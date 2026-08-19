"""Run inference for all 12 Task 2 combos and assemble the submission.

For each (modality x target field) it:
    1. picks the checkpoint (DA if present, else best pretrain)
    2. runs inference_fase5 on Validating_prospective/<mod>/0.1T with
       EMA + flip TTA + histogram matching to the target-field distribution
    3. writes predictions to <inference_dir>/<mod>/<target>/

Then optionally runs SynthSeg on the predictions and build_submission.

Why histogram matching is on by default: the network is trained on synthetic
degradation, so on REAL 0.1T its output sits at a lower intensity than the true
target field (we measured pred ~ 0.77 x reference). nRMSE and SSIM are both
sensitive to that global offset, and matching is a monotone map so it cannot
change anatomy - only the intensity distribution.

    # see the plan
    python scripts/run_submission.py --dry_run

    # inference for everything
    python scripts/run_submission.py \
        --data_root /data/juan/MRIxFields2026_data \
        --runs_root /data/juan/runs \
        --bbox /data/juan/preprocessed/bbox.json \
        --inference_dir /data/juan/inference

    # compare DA vs pretrain-only for one modality before committing
    python scripts/run_submission.py --only T1W --variant both
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"

TARGETS = ["1.5T", "3T", "5T", "7T"]
MODALITIES = ["T1W", "T2W", "T2FLAIR"]


def best_checkpoint(run_dir: Path, prefix: str):
    """Highest-validation-SSIM checkpoint, falling back to <prefix>_last.pth."""
    import torch
    if not run_dir.is_dir():
        return None, None
    best, best_ssim = None, -1.0
    for p in sorted(run_dir.glob(f"{prefix}_ep*.pth")):
        try:
            v = torch.load(p, map_location="cpu").get("val") or {}
            s = v.get("ssim")
            if s is not None and s == s and s > best_ssim:
                best, best_ssim = p, s
        except Exception:
            continue
    if best is not None:
        return best, best_ssim
    last = run_dir / f"{prefix}_last.pth"
    return (last, None) if last.exists() else (None, None)


def resolve_model(runs_root: Path, target: str, mod: str, variant: str):
    """Return (checkpoint, config_path, tag) for the requested variant."""
    da_dir = runs_root / f"task2_0.1T_to_{target}_{mod}_da" / "nafnet_semi"
    pre_dir = runs_root / f"task2_0.1T_to_{target}_{mod}_interim" / "nafnet"
    cfg_da = REPO / "configs/task2/sweep/da" / f"0.1T_to_{target}_{mod}.yaml"
    cfg_pre = REPO / "configs/task2/sweep/interim" / f"0.1T_to_{target}_{mod}.yaml"

    if variant in ("da", "auto"):
        ck, ssim = best_checkpoint(da_dir, "semi")
        if ck is not None:
            return ck, cfg_da, "da", ssim
        if variant == "da":
            return None, None, "da", None
    ck, ssim = best_checkpoint(pre_dir, "pretrain")
    return ck, cfg_pre, "pretrain", ssim


def main():
    ap = argparse.ArgumentParser(description="Task 2 inference + submission assembly")
    ap.add_argument("--data_root", default=None, help="default: $DATA_DIR")
    ap.add_argument("--runs_root", default=None, help="default: $OUTPUT_DIR")
    ap.add_argument("--inference_dir", default=None, help="default: $INFERENCE_DIR")
    ap.add_argument("--bbox", default=None)
    ap.add_argument("--split", default="Validating_prospective")
    ap.add_argument("--variant", choices=["auto", "da", "pretrain", "both"], default="auto",
                    help="auto=DA when available; both=write da/ and pretrain/ side by side "
                         "so you can compare rank-sums before choosing")
    ap.add_argument("--only", default=None, help="subset: 'T1W' or 'T1W:3T,T2W:1.5T'")
    ap.add_argument("--views", default="axial")
    ap.add_argument("--no_tta", action="store_true")
    ap.add_argument("--no_match_hist", action="store_true",
                    help="disable histogram matching (NOT recommended: predictions "
                         "sit ~23%% below the true target intensity)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO))
    from mrixfields.env import load_env, get_data_dir, get_output_dir
    load_env()
    import os
    data_root = Path(args.data_root or get_data_dir())
    runs_root = Path(args.runs_root or get_output_dir())
    inf_dir = Path(args.inference_dir or os.environ.get("INFERENCE_DIR", REPO / "inference"))

    combos = [(t, m) for t in TARGETS for m in MODALITIES]
    if args.only:
        want = set()
        for tok in args.only.split(","):
            if ":" in tok:
                m, t = tok.split(":"); want.add((t, m))
            else:
                want.update((t, tok) for t in TARGETS)   # whole modality
        combos = [c for c in combos if c in want]

    variants = ["da", "pretrain"] if args.variant == "both" else [args.variant]

    print(f"data_root  = {data_root}\nruns_root  = {runs_root}\ninference  = {inf_dir}\n"
          f"combos     = {len(combos)}  variants = {variants}\n")

    missing, ok = [], []
    for target, mod in combos:
        in_dir = data_root / args.split / mod / "0.1T"
        if not in_dir.is_dir():
            print(f"[skip] {mod}:{target} -- no input dir {in_dir}")
            missing.append((mod, target, "no input"))
            continue

        for var in variants:
            ck, cfg, tag, ssim = resolve_model(runs_root, target, mod, var)
            if ck is None:
                print(f"[skip] {mod}:{target} ({var}) -- no checkpoint")
                missing.append((mod, target, f"no {var} checkpoint"))
                continue

            out = inf_dir / (tag if args.variant == "both" else "") / mod / target
            s = f" (val SSIM {ssim:.4f})" if ssim else ""
            print(f"[{mod}:{target}] {tag}{s}\n    ckpt {ck.name}\n    -> {out}")

            cmd = [sys.executable, str(SCRIPTS / "inference_fase5.py"),
                   "--config", str(cfg), "--checkpoints", str(ck),
                   "--input_dir", str(in_dir), "--output_dir", str(out),
                   "--use_ema", "--views", args.views]
            if args.bbox:
                cmd += ["--bbox", args.bbox]
            if not args.no_tta:
                cmd.append("--tta")
            if not args.no_match_hist:
                # match to the REAL target-field distribution of this modality
                ref = data_root / "Training_retrospective" / mod / target
                if ref.is_dir():
                    cmd += ["--match_hist", str(ref)]
                else:
                    print(f"    [warn] no reference dir {ref}; histogram matching OFF")

            print("    $ " + " ".join(cmd))
            if not args.dry_run:
                r = subprocess.run(cmd, cwd=str(REPO))
                if r.returncode != 0:
                    print(f"    !! inference FAILED for {mod}:{target} ({tag})")
                    missing.append((mod, target, f"{tag} inference failed"))
                    continue
            ok.append((mod, target, tag))

    print(f"\n{'='*70}\nDone: {len(ok)} inference run(s).")
    if missing:
        print("MISSING / FAILED:")
        for m, t, why in missing:
            print(f"  {m}:{t}  -- {why}")
        print("\nA missing combo means an INCOMPLETE Task 2 submission. Resolve these\n"
              "before packaging.")
    print(f"{'='*70}")
    print("\nNext:\n"
          "  1. QC:      python scripts/qc_visual.py --input_dir <0.1T> --pred_dir <out> --out_dir qc/\n"
          "  2. Segment: python Evaluation/segment.py --input_dir <pred> --output_dir <pred_seg>\n"
          "  3. Package: python Submission/build_submission/build_submission.py --tasks task2 \\\n"
          "                  --predictions-dir <inference_dir> --predictions-seg-dir <seg_dir> \\\n"
          "                  --output-dir <submission_dir>")


if __name__ == "__main__":
    main()
