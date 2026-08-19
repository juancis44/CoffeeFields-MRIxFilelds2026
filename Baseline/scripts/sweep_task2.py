"""Propagate the validated Task 2 recipe to all (target x modality) combos.

The recipe validated on 0.1T->3T T1W is, per combo:
    1. extract slices (retro_train: 0.1T + target)          [cheap, --skip_existing]
    2. INTERIM pretrain  (train_fase2, synthetic pairs only, synthetic validation)
    3. pick the best pretrain checkpoint by validation SSIM
    4. DA / Fase 4       (train_fase4, synthetic anchor + real-0.1T MIND, GAN off)
       warm-started from that checkpoint

Everything here is pairing-independent, so it is unaffected by the broken
`Validating_prospective` split.

    # see the plan without running anything
    python scripts/sweep_task2.py --dry_run

    # run everything still missing (skips combos whose DA checkpoint exists)
    nohup python scripts/sweep_task2.py > logs/sweep.log 2>&1 &

    # only a subset, 2 GPUs
    python scripts/sweep_task2.py --only T1W:1.5T,T2W:3T --nproc 2
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent          # Baseline/
SCRIPTS = REPO / "scripts"

# recoverable field gaps first; T1W first (most validated)
TARGET_ORDER = ["1.5T", "3T", "5T", "7T"]
MODALITY_ORDER = ["T1W", "T2W", "T2FLAIR"]

MODEL = dict(input_nc=5, output_nc=1, width=32,
             enc_blk_nums=[2, 2, 4], middle_blk_num=6, dec_blk_nums=[2, 2, 2],
             global_residual=True, drop=0.0)
DISC = dict(ndf=64, n_layers=3, gan_mode="lsgan")


# --------------------------------------------------------------------------- #
#  Config generation (mirrors the validated 0.1T->3T T1W configs)
# --------------------------------------------------------------------------- #
def interim_cfg(target, modality, epochs, bs=8, lr_scale=1.0, max_subj=0, degrade_preset="default"):
    return dict(
        task=2, task_name=f"task2_0.1T_to_{target}_{modality}_interim", method="nafnet",
        model=MODEL,
        data=dict(modality=modality, source_field="0.1T", target_field=target,
                  n_neighbors=2, val_source="synthetic", synth_val_holdout=5,
                  max_train_subjects=max_subj, degrade_preset=degrade_preset),
        loss=dict(l1=1.0, ssim=1.0, edge=0.1, lpips=0.1),
        train=dict(batch_size=bs, num_workers=8, cache=True, amp=True, ema_decay=0.999,
                   wd=1.0e-4, grad_clip=1.0, print_every=100, val_every=2, val_lpips=True,
                   pretrain_epochs=epochs, pretrain_lr=2.0e-4 * lr_scale,
                   finetune_epochs=0, finetune_lr=1.0e-4),
    )


def da_cfg(target, modality, epochs, bs=8, lr_scale=1.0, max_subj=0, degrade_preset="default"):
    return dict(
        task=2, task_name=f"task2_0.1T_to_{target}_{modality}_da", method="nafnet_semi",
        model=MODEL, disc=DISC,
        data=dict(modality=modality, source_field="0.1T", target_field=target,
                  n_neighbors=2, paired_source="synthetic",
                  val_source="synthetic", synth_val_holdout=5,
                  max_train_subjects=max_subj, degrade_preset=degrade_preset),
        loss=dict(l1=1.0, ssim=1.0, edge=0.1, lpips=0.1,
                  adv=0.0, mind=0.5, mind_radius=2,
                  adv_unpaired=0.0, mind_unpaired=0.5),
        train=dict(supervised=True, batch_size=bs, num_workers=8, cache=True, amp=True,
                   ema_decay=0.999, wd=1.0e-4, grad_clip=1.0, print_every=100,
                   val_every=2, val_lpips=True,
                   finetune_epochs=epochs, finetune_lr=3.0e-5 * lr_scale, disc_lr=1.0e-5),
    )


def write_cfg(cfg, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def best_checkpoint(run_dir, prefix):
    """Checkpoint with the highest validation SSIM; falls back to <prefix>_last.pth."""
    import torch
    run_dir = Path(run_dir)
    best, best_ssim = None, -1.0
    for p in sorted(run_dir.glob(f"{prefix}_ep*.pth")):
        try:
            v = torch.load(p, map_location="cpu").get("val") or {}
            s = v.get("ssim")
            if s is not None and s == s and s > best_ssim:   # skip NaN
                best, best_ssim = p, s
        except Exception:
            continue
    if best is not None:
        print(f"    best {prefix}: {best.name} (SSIM={best_ssim:.4f})", flush=True)
        return best
    last = run_dir / f"{prefix}_last.pth"
    return last if last.exists() else None


def run(cmd, log_path=None, dry=False):
    pretty = " ".join(str(c) for c in cmd)
    print(f"    $ {pretty}", flush=True)
    if dry:
        return True
    argv = [str(c) for c in cmd]
    if log_path:
        with open(log_path, "a") as lf:
            lf.write(f"\n$ {pretty}\n"); lf.flush()
            r = subprocess.run(argv, cwd=str(REPO), stdout=lf, stderr=subprocess.STDOUT)
    else:
        r = subprocess.run(argv, cwd=str(REPO))
    return r.returncode == 0


def train_cmd(script, cfg_path, args, extra=()):
    """Training always goes through torchrun (DDP-capable, also fine with 1 proc)."""
    cmd = ["torchrun", f"--nproc_per_node={args.nproc}",
           str(SCRIPTS / script), "--config", str(cfg_path),
           "--preprocessed_dir", args.preprocessed_dir, "--bbox", args.bbox]
    return cmd + [str(e) for e in extra]


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Task 2 sweep: propagate the recipe to all combos")
    ap.add_argument("--preprocessed_dir", required=True)
    ap.add_argument("--bbox", required=True)
    ap.add_argument("--out_root", default=None, help="runs dir (default: $OUTPUT_DIR from .env)")
    ap.add_argument("--data_dir", default=None, help="raw data root (for extract-slices)")
    ap.add_argument("--cfg_root", default=None, help="where to write generated configs")
    ap.add_argument("--only", default=None, help="subset like 'T1W:1.5T,T2W:3T'")
    ap.add_argument("--targets", default=",".join(TARGET_ORDER))
    ap.add_argument("--modalities", default=",".join(MODALITY_ORDER))
    ap.add_argument("--pretrain_epochs", type=int, default=40)
    ap.add_argument("--da_epochs", type=int, default=30)
    ap.add_argument("--max_train_subjects", type=int, default=0,
                    help="cap synthetic training subjects per epoch (0=all). Biggest wall-clock lever")
    ap.add_argument("--batch_size", type=int, default=8,
                    help="per-GPU batch for the pretrain stage")
    ap.add_argument("--da_batch_size", type=int, default=0,
                    help="per-GPU batch for the DA stage (0 = half of --batch_size). "
                         "DA runs two forward passes per step (synthetic + unpaired real) "
                         "plus the discriminator, so it needs ~2x the memory of pretrain.")
    ap.add_argument("--nproc", type=int, default=2,
                    help="GPUs per run; training always uses torchrun")
    ap.add_argument("--stage", choices=["all", "pretrain", "da"], default="all")
    ap.add_argument("--redo", action="store_true", help="rerun combos already finished")
    ap.add_argument("--no_extract", action="store_true", help="skip the extract-slices step")
    ap.add_argument("--degrade_preset", default="default",
                    choices=["default", "lowfreq", "realistic", "physical"],
                    help="low-field simulator: 'lowfreq' matches real 0.1T spectrum "
                         "(fixes denoising-vs-superres). Requires re-training.")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    # sqrt LR scaling: safer than linear given the instability we hit earlier
    lr_scale = (args.batch_size / 8.0) ** 0.5
    da_bs = args.da_batch_size or max(1, args.batch_size // 2)
    da_lr_scale = (da_bs / 8.0) ** 0.5

    sys.path.insert(0, str(REPO))
    from mrixfields.env import load_env, get_output_dir, get_data_dir
    load_env()
    out_root = Path(args.out_root or get_output_dir())
    data_dir = args.data_dir or get_data_dir()
    cfg_root = Path(args.cfg_root or (REPO / "configs" / "task2" / "sweep"))
    logs = REPO / "logs" / "sweep"; logs.mkdir(parents=True, exist_ok=True)

    # build the combo list, in priority order
    targets = [t for t in args.targets.split(",") if t]
    mods = [m for m in args.modalities.split(",") if m]
    combos = [(t, m) for t in targets for m in mods]
    if args.only:
        want = {tuple(x.split(":")[::-1]) for x in args.only.split(",")}  # 'MOD:TGT' -> (TGT,MOD)
        combos = [c for c in combos if c in want]

    print(f"Sweep over {len(combos)} combo(s). out_root={out_root}\n"
          f"pretrain_epochs={args.pretrain_epochs} da_epochs={args.da_epochs} nproc={args.nproc} "
          f"batch={args.batch_size}/GPU pretrain, {da_bs}/GPU DA "
          f"(eff {args.batch_size*args.nproc}/{da_bs*args.nproc}) lr_scale={lr_scale:.2f}\n")

    todo, done = [], []
    for target, mod in combos:
        da_last = out_root / f"task2_0.1T_to_{target}_{mod}_da" / "nafnet_semi" / "semi_last.pth"
        (done if (da_last.exists() and not args.redo) else todo).append((target, mod))
    if done:
        print("Already finished (skipping): " + ", ".join(f"{m}:{t}" for t, m in done) + "\n")
    if not todo:
        print("Nothing to do."); return

    t_start = time.time()
    for i, (target, mod) in enumerate(todo, 1):
        tag = f"{mod}_0.1T_to_{target}"
        print(f"\n{'='*70}\n[{i}/{len(todo)}] {tag}\n{'='*70}", flush=True)
        log = logs / f"{tag}.log"

        # 1. slices (cheap when already extracted)
        if not args.no_extract:
            ecmd = [sys.executable, SCRIPTS / "preprocess.py", "extract-slices",
                    "--splits", "retro_train", "--modalities", mod,
                    "--fields", "0.1T", target, "--skip_existing",
                    "--data_dir", data_dir, "--output_dir", args.preprocessed_dir]
            if args.max_train_subjects > 0:
                # only extract what we will actually train on (bounds disk usage)
                ecmd += ["--max_subjects", str(args.max_train_subjects + 5)]  # +5 for synth val holdout
            ok = run(ecmd, log, args.dry_run)
            if not ok:
                print(f"  !! extract-slices failed for {tag}; skipping combo"); continue

        # 2. interim pretrain
        icfg = write_cfg(interim_cfg(target, mod, args.pretrain_epochs, args.batch_size, lr_scale, args.max_train_subjects, args.degrade_preset),
                         cfg_root / "interim" / f"0.1T_to_{target}_{mod}.yaml")
        idir = out_root / f"task2_0.1T_to_{target}_{mod}_interim" / "nafnet"
        if args.stage in ("all", "pretrain"):
            if (idir / "pretrain_last.pth").exists() and not args.redo:
                print("    pretrain already present, skipping")
            else:
                if not run(train_cmd("train_fase2.py", icfg, args), log, args.dry_run):
                    print(f"  !! pretrain failed for {tag}; see {log}", flush=True)
                    if i == 1:
                        # first combo failing = config/env problem, not data-specific.
                        # Abort instead of repeating the same failure 11 more times.
                        print("  ABORTING sweep: the first combo failed, so this is almost\n"
                              "  certainly a config/environment issue (OOM, bad path, missing\n"
                              f"  slices). Inspect: tail -40 {log}", flush=True)
                        return
                    continue

        # 3. best pretrain checkpoint
        init = None
        if args.stage in ("all", "da"):
            init = None if args.dry_run else best_checkpoint(idir, "pretrain")
            if init is None and not args.dry_run:
                print(f"  !! no pretrain checkpoint for {tag}; skipping DA"); continue

        # 4. DA (Fase 4)
        if args.stage in ("all", "da"):
            dcfg = write_cfg(da_cfg(target, mod, args.da_epochs, da_bs, da_lr_scale, args.max_train_subjects, args.degrade_preset), 
                             cfg_root / "da" / f"0.1T_to_{target}_{mod}.yaml")
            extra = ["--init_from", init, "--use_ema"] if init else []
            if not run(train_cmd("train_fase4.py", dcfg, args, extra), log, args.dry_run):
                print(f"  !! DA failed for {tag}; see {log}", flush=True)
                if i == 1:
                    print("  ABORTING sweep: DA failed on the FIRST combo, so it is a config/\n"
                          "  environment issue (OOM, disk full), not data-specific. The pretrain\n"
                          "  checkpoints are kept and are usable on their own.\n"
                          f"  Inspect: tail -40 {log}", flush=True)
                    return
                continue

        print(f"  done {tag}  (elapsed {(time.time()-t_start)/3600:.1f} h)", flush=True)

    print(f"\nSweep finished in {(time.time()-t_start)/3600:.1f} h. Logs: {logs}")


if __name__ == "__main__":
    main()
