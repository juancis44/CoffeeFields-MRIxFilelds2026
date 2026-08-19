"""Generate the full Task 2 config sweep: 4 targets x 3 modalities per method.

Task 2 allows up to 12 models (4 target fields x 3 modalities). This writes one
config per (target, modality) for each requested method, based on the same
templates as the 0.1T_to_3T_T1W configs.

    python scripts/generate_configs_task2.py                 # all methods
    python scripts/generate_configs_task2.py --methods nafnet nafnet_gan

Outputs to configs/task2/<method>/0.1T_to_<target>_<modality>.yaml
"""

import argparse
from pathlib import Path

import yaml

TARGETS = ["1.5T", "3T", "5T", "7T"]
MODALITIES = ["T1W", "T2W", "T2FLAIR"]

MODEL = dict(input_nc=5, output_nc=1, width=32,
             enc_blk_nums=[2, 2, 4], middle_blk_num=6, dec_blk_nums=[2, 2, 2],
             global_residual=True, drop=0.0)
DISC = dict(ndf=64, n_layers=3, gan_mode="lsgan")


def base_train(**over):
    t = dict(batch_size=8, num_workers=8, cache=True, amp=True, ema_decay=0.999,
             wd=1.0e-4, print_every=100, val_every=5, val_lpips=True)
    t.update(over)
    return t


def make_cfg(method, target, modality):
    name = f"task2_0.1T_to_{target}_{modality}"
    data = dict(modality=modality, source_field="0.1T", target_field=target, n_neighbors=2, val_holdout=1, paired_split="pro_val", pair_by="order")
    if method == "nafnet":
        return dict(task=2, task_name=name, method=method, model=MODEL,
                    data=data,
                    loss=dict(l1=1.0, ssim=1.0, edge=0.1, lpips=0.1),
                    train=base_train(pretrain_epochs=30, pretrain_lr=2e-4,
                                     finetune_epochs=100, finetune_lr=1e-4))
    if method == "nafnet_gan":
        return dict(task=2, task_name=name, method=method, model=MODEL, disc=DISC,
                    data=data,
                    loss=dict(l1=1.0, ssim=1.0, edge=0.1, lpips=0.1, adv=0.05,
                              mind=0.5, mind_radius=2),
                    train=base_train(pretrain_epochs=0, finetune_epochs=60,
                                     finetune_lr=5e-5, disc_lr=1e-4))
    if method == "nafnet_semi":
        return dict(task=2, task_name=name, method=method, model=MODEL, disc=DISC,
                    data=data,
                    loss=dict(l1=1.0, ssim=1.0, edge=0.1, lpips=0.1, adv=0.05,
                              mind=0.5, mind_radius=2, adv_unpaired=0.05, mind_unpaired=0.5),
                    train=base_train(finetune_epochs=40, finetune_lr=5e-5, disc_lr=1e-4))
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["nafnet", "nafnet_gan", "nafnet_semi"])
    ap.add_argument("--out_root", default=None, help="default: configs/task2")
    args = ap.parse_args()

    root = Path(args.out_root) if args.out_root else Path(__file__).resolve().parent.parent / "configs" / "task2"
    n = 0
    for method in args.methods:
        d = root / method
        d.mkdir(parents=True, exist_ok=True)
        for target in TARGETS:
            for mod in MODALITIES:
                cfg = make_cfg(method, target, mod)
                p = d / f"0.1T_to_{target}_{mod}.yaml"
                p.write_text(yaml.safe_dump(cfg, sort_keys=False))
                n += 1
    print(f"Wrote {n} configs under {root} ({len(args.methods)} methods x {len(TARGETS)}x{len(MODALITIES)})")


if __name__ == "__main__":
    main()
