"""Shared training utilities for Fase 2 / Fase 3 (Task 2).

Keeps DDP setup, validation metrics, and dataloader construction in one place so
the phase trainers stay small and consistent.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .data.paired25d import RAMCachedPaired25D, SyntheticLowfieldPaired25D, Unpaired25D
from .data.degrade import (DegradeConfig, lowfreq_preset,
                           realistic_preset, physical_preset)


def cycle(loader):
    """Infinite iterator over a DataLoader (for pairing loaders of unequal length)."""
    while True:
        for b in loader:
            yield b


# --------------------------------------------------------------------------- #
#  DDP
# --------------------------------------------------------------------------- #
def ddp_setup():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        return True, dist.get_rank(), dist.get_world_size(), local
    return False, 0, 1, 0


def is_main(rank: int) -> bool:
    return rank == 0


# --------------------------------------------------------------------------- #
#  Validation metrics (the 3 voxel metrics used by the ranking)
# --------------------------------------------------------------------------- #
def nrmse(pred, tgt):
    rng = tgt.max() - tgt.min()
    return float(np.sqrt(np.mean((pred - tgt) ** 2)) / (rng + 1e-8))


def ssim_np(pred, tgt):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(pred, tgt, data_range=1.0))


def make_lpips(device, enabled: bool = True):
    """Return a grayscale-friendly LPIPS callable, or None if disabled/unavailable."""
    if not enabled:
        return None
    try:
        import lpips as _lpips
        base = _lpips.LPIPS(net="alex").to(device).eval()
        return lambda a, b: base(a.repeat(1, 3, 1, 1), b.repeat(1, 3, 1, 1))
    except Exception as e:  # offline / weights missing -> skip LPIPS metric
        print(f"[warn] LPIPS disabled ({e}); validating on nRMSE/SSIM only.", flush=True)
        return None


@torch.no_grad()
def validate(model, loader, device, lpips_fn=None):
    model.eval()
    ns, ss, lp, n = 0.0, 0.0, 0.0, 0
    for batch in loader:
        src = batch["source"].to(device)
        tgt = batch["target"].to(device)
        pred = model(src).clamp(-1, 1)
        p01 = ((pred + 1) * 0.5).float().cpu().numpy()
        t01 = ((tgt + 1) * 0.5).float().cpu().numpy()
        for b in range(p01.shape[0]):
            tb = t01[b, 0]
            if (tb.max() - tb.min()) < 0.05:   # near-empty slice: no signal, skip
                continue
            ns += nrmse(p01[b, 0], tb)
            ss += ssim_np(p01[b, 0], tb)
            n += 1
        if lpips_fn is not None:
            lp += float(lpips_fn(pred, tgt).mean()) * p01.shape[0]
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    lp_val = (lp / n) if lpips_fn is not None else float("nan")
    return ns / n, ss / n, lp_val


# --------------------------------------------------------------------------- #
#  Dataloaders
# --------------------------------------------------------------------------- #
def _holdout_subjects(prep, cfg):
    """(train_subjects, val_subjects) held out from pro_train, or (None, None).

    Used when the challenge validation split isn't available locally: set
    data.val_holdout: K to reserve K prospective subjects for validation.
    """
    k = int(cfg["data"].get("val_holdout", 0))
    if k <= 0:
        return None, None
    from .data.paired25d import parse_npz_name
    d = cfg["data"]
    split = d.get("paired_split", "pro_train")
    base = Path(prep) / split / d["modality"] / d["source_field"]
    subs = sorted({parse_npz_name(f)[0] for f in base.glob("*.npz")})
    if len(subs) <= k:
        return None, None
    return subs[:-k], subs[-k:]


def _synth_holdout(prep, cfg):
    """(train_retro, val_retro) subjects from retro_train target field for
    synthetic validation, or (None, None). Set data.synth_val_holdout: K."""
    k = int(cfg["data"].get("synth_val_holdout", 0))
    if k <= 0:
        return None, None
    from .data.paired25d import parse_npz_name
    d = cfg["data"]
    base = Path(prep) / "retro_train" / d["modality"] / d["target_field"]
    subs = sorted({parse_npz_name(f)[0] for f in base.glob("*.npz")})
    if len(subs) <= k:
        return None, None
    train, val = subs[:-k], subs[-k:]
    cap = int(d.get("max_train_subjects", 0))
    if cap > 0:
        train = train[:cap]        # cap synthetic training subjects (wall-clock lever)
    return train, val


def _degrade_cfg(cfg):
    """Pick the low-field simulator preset from data.degrade_preset.
    'lowfreq' matches the spectrum of real 0.1T (noise before blur, heavier
    blur/downsample) so the net learns super-resolution, not denoising.
    Default 'default' reproduces the originally submitted models."""
    preset = str(cfg["data"].get("degrade_preset", "default")).lower()
    builder = {"lowfreq": lowfreq_preset, "low_freq": lowfreq_preset,
               "spectrum": lowfreq_preset, "realistic": realistic_preset,
               "physical": physical_preset}.get(preset, DegradeConfig)
    dcfg = builder()

    # Attach the REAL 0.1T intensity distribution so the simulator can map
    # high-field contrast onto it (the relaxometric difference blur+noise cannot
    # reproduce). Built once per run from the preprocessed source-field slices.
    if preset in ("realistic", "physical") and cfg["data"].get("contrast_map", True):
        dcfg.contrast_ref = _build_contrast_ref(_PREP_DIR, cfg)
    return dcfg


_CONTRAST_CACHE = {}
_PREP_DIR = None


def _build_contrast_ref(prep, cfg, n_subjects=12, max_slices=40):
    """Population quantile function of REAL source-field (0.1T) brain intensities."""
    if prep is None:
        print("[warn] no preprocessed dir for contrast_ref; contrast map disabled", flush=True)
        return None
    d = cfg["data"]
    key = (str(prep), d["modality"], d["source_field"])
    if key in _CONTRAST_CACHE:
        return _CONTRAST_CACHE[key]
    from .data.paired25d import parse_npz_name
    base = Path(prep) / "retro_train" / d["modality"] / d["source_field"]
    files = sorted(base.glob("*.npz"))
    if not files:
        print(f"[warn] no real {d['source_field']} slices under {base}; "
              f"contrast map disabled", flush=True)
        _CONTRAST_CACHE[key] = None
        return None
    subs = sorted({parse_npz_name(f)[0] for f in files})[:n_subjects]
    pool = []
    for subj in subs:
        sf = [f for f in files if parse_npz_name(f)[0] == subj]
        lo, hi = int(len(sf) * 0.3), int(len(sf) * 0.7)
        for f in sorted(sf)[lo:hi][:max_slices]:
            img = np.load(f)["image"].astype(np.float32)
            b = img[img > 0.05]
            if b.size:
                pool.append(b)
    if not pool:
        _CONTRAST_CACHE[key] = None
        return None
    pool = np.concatenate(pool)
    if pool.size > 4_000_000:
        pool = np.random.default_rng(0).choice(pool, 4_000_000, replace=False)
    q = np.linspace(0.0, 1.0, 256)
    ref = (q, np.quantile(pool, q).astype(np.float32))
    print(f"[degrade] contrast_ref from {len(subs)} real {d['source_field']} subjects "
          f"({pool.size} voxels): brain mean {pool.mean():.3f}", flush=True)
    _CONTRAST_CACHE[key] = ref
    return ref


def build_train_loader(cfg, prep, bbox, world, rank, phase):
    global _PREP_DIR
    _PREP_DIR = prep
    d = cfg["data"]; t = cfg["train"]
    K = d["n_neighbors"]; mod = d["modality"]
    src_f, tgt_f = d["source_field"], d["target_field"]

    if phase == "pretrain":
        train_synth, _ = _synth_holdout(prep, cfg)
        ds = SyntheticLowfieldPaired25D(
            prep, "retro_train", mod, target_field=tgt_f, bbox=bbox,
            n_neighbors=K, augment=True, cache=t.get("cache", True),
            degrade_cfg=_degrade_cfg(cfg), subjects=train_synth,
        )
    elif d.get("paired_source", "real") == "synthetic":
        # supervised branch anchored on CORRECT synthetic pairs (real pairs unusable)
        train_synth, _ = _synth_holdout(prep, cfg)
        ds = SyntheticLowfieldPaired25D(
            prep, "retro_train", mod, target_field=tgt_f, bbox=bbox,
            n_neighbors=K, augment=True, cache=t.get("cache", True),
            degrade_cfg=_degrade_cfg(cfg), subjects=train_synth,
        )
    else:
        train_subs, _ = _holdout_subjects(prep, cfg)
        paired_split = d.get("paired_split", "pro_train")
        ds = RAMCachedPaired25D(
            prep, paired_split, mod, src_f, tgt_f, bbox=bbox,
            n_neighbors=K, augment=True, cache=t.get("cache", True),
            subjects=train_subs, pair_by=d.get("pair_by", "id"),
        )

    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True) if world > 1 else None
    loader = DataLoader(
        ds, batch_size=t["batch_size"], shuffle=(sampler is None), sampler=sampler,
        num_workers=t.get("num_workers", 8), pin_memory=True, drop_last=True,
        persistent_workers=t.get("num_workers", 8) > 0,
    )
    return ds, loader, sampler


def build_unpaired_loader(cfg, prep, bbox, world, rank, field, n_neighbors):
    """Single-field unpaired loader from retro_train (Fase 4)."""
    d = cfg["data"]; t = cfg["train"]
    ds = Unpaired25D(
        prep, "retro_train", d["modality"], field, bbox=bbox,
        n_neighbors=n_neighbors, augment=True, cache=t.get("cache", True),
    )
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True) if world > 1 else None
    loader = DataLoader(
        ds, batch_size=t["batch_size"], shuffle=(sampler is None), sampler=sampler,
        num_workers=t.get("num_workers", 8), pin_memory=True, drop_last=True,
        persistent_workers=t.get("num_workers", 8) > 0,
    )
    return ds, loader, sampler


def build_val_loader(cfg, prep, bbox):
    global _PREP_DIR
    _PREP_DIR = prep
    d = cfg["data"]
    val_source = d.get("val_source", "paired")
    if val_source == "none":
        print("[val] disabled (val_source: none).", flush=True)
        return None
    if val_source == "synthetic":
        _, val_synth = _synth_holdout(prep, cfg)
        if val_synth is None:
            print("[warn] val_source=synthetic but synth_val_holdout unset/too large; "
                  "no validation.", flush=True)
            return None
        ds = SyntheticLowfieldPaired25D(
            prep, "retro_train", d["modality"], target_field=d["target_field"],
            bbox=bbox, n_neighbors=d["n_neighbors"], augment=False, cache=True,
            degrade_cfg=_degrade_cfg(cfg), subjects=val_synth)
        print(f"[val] synthetic pairs from {len(val_synth)} held-out retro subjects "
              f"(correctly paired).", flush=True)
        return DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                          num_workers=4, pin_memory=True)
    _, val_subs = _holdout_subjects(prep, cfg)
    try:
        if val_subs is not None:
            ds = RAMCachedPaired25D(
                prep, d.get("paired_split", "pro_train"), d["modality"],
                d["source_field"], d["target_field"],
                bbox=bbox, n_neighbors=d["n_neighbors"], augment=False, cache=True,
                subjects=val_subs, pair_by=d.get("pair_by", "id"))
            print(f"[val] holdout of {len(val_subs)} pro_train subjects: {val_subs}", flush=True)
        else:
            ds = RAMCachedPaired25D(
                prep, "pro_val", d["modality"], d["source_field"], d["target_field"],
                bbox=bbox, n_neighbors=d["n_neighbors"], augment=False, cache=True,
                pair_by=d.get("pair_by", "id"))
    except (FileNotFoundError, ValueError) as e:
        print(f"[warn] no validation set ({e}); training WITHOUT periodic validation. "
              f"Set data.val_holdout: K to validate on a pro_train holdout.", flush=True)
        return None
    return DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                      num_workers=4, pin_memory=True)
