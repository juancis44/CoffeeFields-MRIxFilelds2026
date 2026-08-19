"""Shared helpers for the per-phase smoke tests.

The smoke tests fabricate a *tiny* fake dataset (a handful of small slices) so
you can launch a real 1-2 step training/inference for each phase and confirm the
whole code path runs on your torch/GPU install — without needing the full
challenge dataset. They are NOT accuracy tests; they check that the pipeline
executes end-to-end and produces finite outputs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent          # Baseline/
SCRIPTS = REPO / "scripts"


def brain_slice(H, W, rng):
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = H / 2, W / 2
    m = (((xx - cx) / (W * 0.35)) ** 2 + ((yy - cy) / (H * 0.4)) ** 2) < 1
    img = np.zeros((H, W), np.float32)
    img[m] = 0.6
    img += 0.3 * np.exp(-(((xx - cx) / (W * 0.15)) ** 2 + ((yy - cy) / (H * 0.15)) ** 2))
    img += rng.normal(0, 0.02, (H, W))
    return np.clip(img, 0, 1).astype(np.float32)


def _write_npz_group(base, split, modality, field, subjects, s0, n_slices, H, W, rng):
    d = base / split / modality / field
    d.mkdir(parents=True, exist_ok=True)
    for subj in subjects:
        for s in range(s0, s0 + n_slices):
            img = brain_slice(H, W, rng)
            np.savez_compressed(d / f"{split}_{modality}_{field}_{subj}_s{s:03d}.npz", image=img)


def make_fake_prep(prep: Path, modality="T1W", source="0.1T", target="3T",
                   H=64, W=64, n_slices=8, s0=120, seed=0):
    """Build pro_train / pro_val (paired) + retro_train (target + unpaired source)."""
    rng = np.random.default_rng(seed)
    prep.mkdir(parents=True, exist_ok=True)
    pro = ["P_0001", "P_0002"]
    retro = ["R_0001", "R_0002", "R_0003"]
    for split in ("pro_train", "pro_val"):
        for field in (source, target):
            _write_npz_group(prep, split, modality, field, pro, s0, n_slices, H, W, rng)
    # retrospective high-field (synthetic pretrain + Fase 4 real target)
    _write_npz_group(prep, "retro_train", modality, target, retro, s0, n_slices, H, W, rng)
    # retrospective UNPAIRED 0.1T (Fase 4 semi-supervised source) — different subjects
    _write_npz_group(prep, "retro_train", modality, source,
                     ["R_0101", "R_0102"], s0, n_slices, H, W, rng)

    bbox = {"row": [0, H], "col": [0, W], "slice": [s0, s0 + n_slices],
            "thresh": 0.02, "margin": 0}
    (prep / "bbox.json").write_text(json.dumps(bbox, indent=2))
    return bbox


def make_fake_nifti_inputs(in_dir: Path, modality="T1W", source="0.1T",
                           H=64, W=64, Z=8, seed=1):
    """Create tiny 0.1T .nii.gz volumes to feed inference."""
    import nibabel as nib
    rng = np.random.default_rng(seed)
    d = in_dir / modality / source
    d.mkdir(parents=True, exist_ok=True)
    for subj in ("P_0001", "P_0002"):
        vol = np.stack([brain_slice(H, W, rng) for _ in range(Z)], axis=2)
        nib.save(nib.Nifti1Image(vol.astype(np.float32), np.eye(4)),
                 str(d / f"P_{modality}_{source}_{subj.split('_')[1]}.nii.gz"))
    return d


def make_fake_nifti_pairs(data_dir: Path, split="Training_prospective", modality="T1W",
                          fields=("0.1T", "3T"), H=64, W=64, Z=8, seed=2):
    """Create paired .nii.gz volumes (same subjects across fields) for QC testing."""
    import nibabel as nib
    rng = np.random.default_rng(seed)
    subjects = ["0001", "0002"]
    for subj in subjects:
        base = np.stack([brain_slice(H, W, rng) for _ in range(Z)], axis=2)
        for field in fields:
            d = data_dir / split / modality / field
            d.mkdir(parents=True, exist_ok=True)
            vol = np.clip(base + rng.normal(0, 0.01, base.shape), 0, 1).astype(np.float32)
            nib.save(nib.Nifti1Image(vol, np.eye(4)),
                     str(d / f"P_{modality}_{field}_{subj}.nii.gz"))
    return data_dir


def write_yaml(path: Path, cfg: dict):
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def run_script(script_name: str, args: list, work: Path):
    """Run a script under scripts/ on CPU with tmp env dirs; assert success."""
    env = dict(os.environ)
    env.update({
        "PREPROCESSED_DIR": str(work / "prep"),
        "OUTPUT_DIR": str(work / "out"),
        "INFERENCE_DIR": str(work / "inf"),
        "DEVICE": "cpu",
        "CUDA_VISIBLE_DEVICES": "",
    })
    cmd = [sys.executable, str(SCRIPTS / script_name)] + [str(a) for a in args]
    print(f"\n$ {' '.join(cmd)}")
    r = subprocess.run(cmd, env=env, cwd=str(REPO))
    if r.returncode != 0:
        raise SystemExit(f"FAILED: {script_name} (exit {r.returncode})")
    return r


def base_model_cfg():
    """Tiny NAFNet so smoke tests run in seconds on CPU (depth 2)."""
    return dict(input_nc=3, output_nc=1, width=8,
                enc_blk_nums=[1, 1], middle_blk_num=1, dec_blk_nums=[1, 1],
                global_residual=True, drop=0.0)


def base_train_cfg(**over):
    cfg = dict(batch_size=2, num_workers=0, cache=True, amp=False, ema_decay=0.9,
               wd=1.0e-4, print_every=1, val_every=1, val_lpips=False)
    cfg.update(over)
    return cfg
