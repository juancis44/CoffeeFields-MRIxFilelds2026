"""Smoke test — Fase 2 (supervised NAFNet 2.5D).

Fabricates tiny data, launches a real (2-3 step) pretrain+finetune with
train_fase2.py, then runs inference_fase2.py and checks the output volume.

Run:
    python tests/smoke_fase2.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import _smoke_utils as U


def main():
    work = Path(tempfile.mkdtemp(prefix="smoke2_"))
    prep = work / "prep"
    U.make_fake_prep(prep, H=64, W=64, n_slices=8)

    cfg = dict(
        task=2, task_name="smoke_task2", method="nafnet",
        model=U.base_model_cfg(),
        data=dict(modality="T1W", source_field="0.1T", target_field="3T", n_neighbors=1),
        loss=dict(l1=1.0, ssim=1.0, edge=0.1, lpips=0.0),   # lpips off -> no weight download
        train=U.base_train_cfg(pretrain_epochs=1, pretrain_lr=2e-4,
                               finetune_epochs=1, finetune_lr=1e-4),
    )
    cfg_path = U.write_yaml(work / "smoke_fase2.yaml", cfg)

    # --- launch training (CPU, 3 steps per phase) ---
    U.run_script("train_fase2.py",
                 ["--config", cfg_path, "--preprocessed_dir", prep,
                  "--bbox", prep / "bbox.json", "--max_steps", "3"], work)

    ckpt = work / "out" / "smoke_task2" / "nafnet" / "finetune_last.pth"
    assert ckpt.exists(), f"missing checkpoint {ckpt}"
    print(f"[train_fase2] ok -> {ckpt.name}")

    # --- inference ---
    in_dir = U.make_fake_nifti_inputs(work / "nifti_in")
    U.run_script("inference_fase2.py",
                 ["--config", cfg_path, "--checkpoint", ckpt,
                  "--input_dir", in_dir, "--output_dir", work / "inf",
                  "--bbox", prep / "bbox.json", "--use_ema"], work)

    outs = list((work / "inf").rglob("*.nii.gz"))
    assert outs, "no inference outputs"
    import nibabel as nib
    arr = nib.load(str(outs[0])).get_fdata()
    assert np.isfinite(arr).all() and arr.min() >= 0
    print(f"[inference_fase2] ok -> {len(outs)} volumes, finite, min>=0")

    print("\n===== FASE 2 SMOKE PASSED =====")


if __name__ == "__main__":
    main()
