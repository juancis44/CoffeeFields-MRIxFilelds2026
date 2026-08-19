"""Fase 2 — supervised 2.5D restoration trainer for Task 2 (0.1T -> target).

Pipeline:
    (1) optional PRETRAIN on synthetic pairs (high-field degraded to 0.1T-like),
    (2) FINETUNE on the real 0.1T->target prospective pairs,
    with AMP (bf16), EMA, cosine LR, and single- or multi-GPU (DDP) support.

Validation reports nRMSE / SSIM / LPIPS on the paired validation split (EMA).

Single GPU:
    python scripts/train_fase2.py --config configs/task2/nafnet/0.1T_to_3T_T1W.yaml

Two GPUs (DDP):
    torchrun --nproc_per_node=2 scripts/train_fase2.py \
        --config configs/task2/nafnet/0.1T_to_3T_T1W.yaml
"""

import argparse
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mrixfields.env import load_env, get_preprocessed_dir, get_output_dir
from mrixfields.data.brain_bbox import load_bbox
from mrixfields.models.nafnet import build_nafnet
from mrixfields.models.ema import ModelEMA
from mrixfields.losses.hybrid import HybridRestorationLoss
from mrixfields.train_utils import (
    ddp_setup, is_main, validate, make_lpips, build_train_loader, build_val_loader,
)


def train_phase(phase, cfg, model, ema, criterion, device, prep, bbox,
                ddp, rank, world, out_dir, val_loader, lpips_fn, max_steps=0):
    t = cfg["train"]
    epochs = t[f"{phase}_epochs"]
    if epochs <= 0:
        return
    lr = t[f"{phase}_lr"]

    _, loader, sampler = build_train_loader(cfg, prep, bbox, world, rank, phase)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=t.get("wd", 1e-4))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * len(loader)))
    use_amp = t.get("amp", True) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    step = 0
    total_steps = epochs * len(loader)
    t0 = time.time()
    for ep in range(epochs):
        if sampler is not None:
            sampler.set_epoch(ep)
        model.train()
        for batch in loader:
            src = batch["source"].to(device, non_blocking=True)
            tgt = batch["target"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                pred = model(src)
            loss, logs = criterion(pred.float(), tgt)      # loss in fp32 for stability
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), t.get("grad_clip", 1.0))
            if torch.isfinite(loss) and torch.isfinite(gnorm):
                opt.step()
                ema.update(model)                          # skip corrupting steps
            sched.step()
            step += 1
            if is_main(rank) and step % t.get("print_every", 100) == 0:
                msg = " ".join(f"{k}={v:.4f}" for k, v in logs.items())
                el = time.time() - t0
                sps = step / max(el, 1e-9)
                eta_h = (total_steps - step) / max(sps, 1e-9) / 3600
                print(f"[{phase}] ep {ep} step {step}/{total_steps} lr {sched.get_last_lr()[0]:.2e} | "
                      f"{msg} | {sps:.2f} it/s ETA {eta_h:.1f}h", flush=True)
            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break

        if is_main(rank) and (ep + 1) % t.get("val_every", 5) == 0 and val_loader is not None:
            ns, ss, lp = validate(ema.ema.to(device), val_loader, device, lpips_fn)
            print(f"[{phase}] ep {ep} VAL  nRMSE={ns:.4f}  SSIM={ss:.4f}  LPIPS={lp:.4f}", flush=True)
            torch.save({"ema": ema.state_dict(), "model": (model.module if ddp else model).state_dict(),
                        "cfg": cfg, "phase": phase, "epoch": ep,
                        "val": {"nrmse": ns, "ssim": ss, "lpips": lp}},
                       out_dir / f"{phase}_ep{ep+1}.pth")

    if is_main(rank):
        torch.save({"ema": ema.state_dict(), "model": (model.module if ddp else model).state_dict(),
                    "cfg": cfg, "phase": phase}, out_dir / f"{phase}_last.pth")


def main():
    ap = argparse.ArgumentParser(description="Fase 2 supervised 2.5D trainer")
    ap.add_argument("--config", required=True)
    ap.add_argument("--preprocessed_dir", default=None)
    ap.add_argument("--bbox", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max_steps", type=int, default=0, help="Stop each phase early (smoke tests)")
    load_env()
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ddp, rank, world, local = ddp_setup()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")

    prep = args.preprocessed_dir or get_preprocessed_dir()
    bbox = load_bbox(args.bbox or str(Path(prep) / "bbox.json"))

    out_dir = Path(get_output_dir()) / cfg["task_name"] / "nafnet"
    if is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output: {out_dir}\nbbox crop "
              f"{bbox['row'][1]-bbox['row'][0]}x{bbox['col'][1]-bbox['col'][0]}", flush=True)

    model = build_nafnet(cfg["model"]).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device)["model"])
    if ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local])

    ema = ModelEMA(model, decay=cfg["train"].get("ema_decay", 0.999))
    criterion = HybridRestorationLoss(
        w_l1=cfg["loss"]["l1"], w_ssim=cfg["loss"]["ssim"],
        w_edge=cfg["loss"]["edge"], w_lpips=cfg["loss"]["lpips"],
    ).to(device)

    val_loader = build_val_loader(cfg, prep, bbox) if is_main(rank) else None
    lpips_fn = make_lpips(device, cfg["train"].get("val_lpips", True)) if is_main(rank) else None

    for phase in ("pretrain", "finetune"):
        train_phase(phase, cfg, model, ema, criterion, device, prep, bbox,
                    ddp, rank, world, out_dir, val_loader, lpips_fn, args.max_steps)

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
