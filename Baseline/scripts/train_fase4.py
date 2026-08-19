"""Fase 4 — semi-supervised training for Task 2.

Uses the abundant UNPAIRED retrospective data on top of the Fase 3 recipe:
    * supervised  (paired pro_train): hybrid + MIND + weak adversarial (Fase 3),
    * unsupervised (unpaired retro):
        - real 0.1T inputs (no target GT) -> G(x) pushed toward the real
          high-field distribution by the discriminator (adversarial),
        - MIND( G(x), x_center ) preserves anatomy without ground truth,
        - real high-field slices (retro) serve as extra *real* examples for D.

This improves generalization and contrast robustness by exposing the model to
the 100 real 0.1T subjects that have no paired target, and to many real
high-field slices. Warm-start from a Fase 3 checkpoint.

    python scripts/train_fase4.py \
        --config configs/task2/nafnet_semi/0.1T_to_3T_T1W.yaml \
        --init_from $OUTPUT_DIR/task2_0.1T_to_3T_T1W/nafnet_gan/finetune_last.pth --use_ema

DDP:
    torchrun --nproc_per_node=2 scripts/train_fase4.py --config ... --init_from ... --use_ema
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
from mrixfields.models.networks import NLayerDiscriminator
from mrixfields.models.ema import ModelEMA
from mrixfields.losses.hybrid import HybridRestorationLoss
from mrixfields.losses.mind import MINDLoss
from mrixfields.losses.adversarial import GANLoss
from mrixfields.train_utils import (
    ddp_setup, is_main, validate, make_lpips, cycle,
    build_train_loader, build_unpaired_loader, build_val_loader,
)


def main():
    ap = argparse.ArgumentParser(description="Fase 4 semi-supervised trainer")
    ap.add_argument("--config", required=True)
    ap.add_argument("--init_from", default=None, help="Fase 3 checkpoint to warm-start G")
    ap.add_argument("--use_ema", action="store_true")
    ap.add_argument("--preprocessed_dir", default=None)
    ap.add_argument("--bbox", default=None)
    ap.add_argument("--max_steps", type=int, default=0)
    load_env()
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    t, L = cfg["train"], cfg["loss"]
    K = cfg["data"]["n_neighbors"]

    ddp, rank, world, local = ddp_setup()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")

    prep = args.preprocessed_dir or get_preprocessed_dir()
    bbox = load_bbox(args.bbox or str(Path(prep) / "bbox.json"))
    out_dir = Path(get_output_dir()) / cfg["task_name"] / "nafnet_semi"
    if is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)

    # --- models ---
    G = build_nafnet(cfg["model"]).to(device)
    if args.init_from:
        ck = torch.load(args.init_from, map_location=device)
        G.load_state_dict(ck["ema"] if (args.use_ema and "ema" in ck) else ck["model"])
        if is_main(rank):
            print(f"Warm-started G from {args.init_from}", flush=True)
    dcfg = cfg.get("disc", {})
    D = NLayerDiscriminator(input_nc=1, ndf=dcfg.get("ndf", 64),
                            n_layers=dcfg.get("n_layers", 3)).to(device)
    if ddp:
        G = nn.parallel.DistributedDataParallel(G, device_ids=[local])
        D = nn.parallel.DistributedDataParallel(D, device_ids=[local])

    ema = ModelEMA(G, decay=t.get("ema_decay", 0.999))
    hybrid = HybridRestorationLoss(w_l1=L["l1"], w_ssim=L["ssim"],
                                   w_edge=L["edge"], w_lpips=L["lpips"]).to(device)
    mind = MINDLoss(radius=L.get("mind_radius", 2)).to(device)
    gan = GANLoss(mode=dcfg.get("gan_mode", "lsgan")).to(device)

    optG = torch.optim.AdamW(G.parameters(), lr=t["finetune_lr"], betas=(0.9, 0.99), weight_decay=t.get("wd", 1e-4))
    optD = torch.optim.AdamW(D.parameters(), lr=t.get("disc_lr", t["finetune_lr"]), betas=(0.9, 0.99))

    # --- data ---
    supervised = bool(t.get("supervised", True))   # False -> unpaired-only (ambiguous pairs skipped)
    _, usrc_loader, usrc_sampler = build_unpaired_loader(cfg, prep, bbox, world, rank, cfg["data"]["source_field"], K)
    _, utgt_loader, _ = build_unpaired_loader(cfg, prep, bbox, world, rank, cfg["data"]["target_field"], 0)
    utgt_it = cycle(utgt_loader)
    if supervised:
        _, paired_loader, paired_sampler = build_train_loader(cfg, prep, bbox, world, rank, "finetune")
        usrc_it = cycle(usrc_loader)
        main_loader, main_sampler = paired_loader, paired_sampler
    else:
        main_loader, main_sampler = usrc_loader, usrc_sampler   # iterate the ~100 real 0.1T
        if is_main(rank):
            print(f"[semi] UNPAIRED-ONLY mode: {len(usrc_loader.dataset)} real 0.1T slices, "
                  f"no paired supervision.", flush=True)

    val_loader = build_val_loader(cfg, prep, bbox) if is_main(rank) else None
    lpips_fn = make_lpips(device, t.get("val_lpips", True)) if is_main(rank) else None

    epochs = t["finetune_epochs"]
    sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(optG, T_max=max(1, epochs * len(main_loader)))
    use_amp = t.get("amp", True) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    w_adv, w_mind = L["adv"], L["mind"]
    w_adv_u, w_mind_u = L["adv_unpaired"], L["mind_unpaired"]
    use_gan = (w_adv > 0) or (w_adv_u > 0)
    if is_main(rank):
        mode = ("synthetic-anchor + real-MIND" if (supervised and cfg["data"].get("paired_source")=="synthetic")
                else ("supervised+unpaired" if supervised else "unpaired-only"))
        print(f"[semi] mode={mode} | GAN={'on' if use_gan else 'off'}", flush=True)

    step = 0
    total_steps = epochs * len(main_loader)
    t0 = time.time()
    for ep in range(epochs):
        if main_sampler is not None:
            main_sampler.set_epoch(ep)
        G.train(); D.train()
        for batch in main_loader:
            if supervised:
                src = batch["source"].to(device, non_blocking=True)
                tgt = batch["target"].to(device, non_blocking=True)
                src_u = next(usrc_it)["image"].to(device, non_blocking=True)
            else:
                src_u = batch["image"].to(device, non_blocking=True)   # main loader IS the unpaired 0.1T
            tgt_u = next(utgt_it)["image"].to(device, non_blocking=True)

            # ---- Discriminator (skipped when GAN is off) ----
            loss_D = src_u.new_zeros(())
            if use_gan:
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    with torch.no_grad():
                        fake_u = G(src_u)
                        fake_p = G(src) if supervised else None
                    if supervised:
                        d_real = 0.5 * (gan(D(tgt), True) + gan(D(tgt_u), True))
                        d_fake = 0.5 * (gan(D(fake_p.detach()), False) + gan(D(fake_u.detach()), False))
                    else:
                        d_real = gan(D(tgt_u), True)
                        d_fake = gan(D(fake_u.detach()), False)
                    loss_D = 0.5 * (d_real + d_fake)
                optD.zero_grad(set_to_none=True)
                loss_D.backward()
                optD.step()

            # ---- Generator (loss in fp32 for stability) ----
            loss_sup = torch.zeros((), device=device)
            logs = {}
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                pred = G(src) if supervised else None
                out_u = G(src_u)
                adv_p = gan(D(pred), True) if (supervised and use_gan) else None
                adv_u = gan(D(out_u), True) if use_gan else None
            out_u = out_u.float()
            if supervised:
                pred = pred.float()
                loss_hyb, logs = hybrid(pred, tgt)
                loss_sup = loss_hyb + w_mind * mind(pred, tgt)
                if use_gan:
                    loss_sup = loss_sup + w_adv * adv_p.float()
            center_u = src_u[:, K:K + 1]                 # 0.1T input center slice
            loss_uns = w_mind_u * mind(out_u, center_u)
            if use_gan:
                loss_uns = loss_uns + w_adv_u * adv_u.float()
            loss_G = loss_sup + loss_uns
            optG.zero_grad(set_to_none=True)
            loss_G.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(G.parameters(), t.get("grad_clip", 1.0))
            if torch.isfinite(loss_G) and torch.isfinite(gnorm):
                optG.step()
                ema.update(G)
            sched_G.step()
            step += 1

            if is_main(rank) and step % t.get("print_every", 100) == 0:
                msg = " ".join(f"{k}={v:.4f}" for k, v in logs.items())
                el = time.time() - t0
                sps = step / max(el, 1e-9)
                eta_h = (total_steps - step) / max(sps, 1e-9) / 3600
                print(f"[semi] ep {ep} step {step}/{total_steps} | {msg} "
                      f"G={float(loss_G):.4f} unsup={float(loss_uns):.4f} | "
                      f"{sps:.2f} it/s ETA {eta_h:.1f}h", flush=True)
            if max_steps_reached(max_steps=args.max_steps, step=step):
                break
        if max_steps_reached(max_steps=args.max_steps, step=step):
            break

        if is_main(rank) and (ep + 1) % t.get("val_every", 5) == 0 and val_loader is not None:
            ns, ss, lp = validate(ema.ema.to(device), val_loader, device, lpips_fn)
            print(f"[semi] ep {ep} VAL  nRMSE={ns:.4f}  SSIM={ss:.4f}  LPIPS={lp:.4f}", flush=True)
            torch.save({"ema": ema.state_dict(), "model": (G.module if ddp else G).state_dict(),
                        "cfg": cfg, "phase": "semi", "epoch": ep,
                        "val": {"nrmse": ns, "ssim": ss, "lpips": lp}},
                       out_dir / f"semi_ep{ep+1}.pth")

    if is_main(rank):
        torch.save({"ema": ema.state_dict(), "model": (G.module if ddp else G).state_dict(),
                    "cfg": cfg, "phase": "semi"}, out_dir / "semi_last.pth")
    if ddp:
        dist.destroy_process_group()


def max_steps_reached(max_steps, step):
    return bool(max_steps) and step >= max_steps


if __name__ == "__main__":
    main()
