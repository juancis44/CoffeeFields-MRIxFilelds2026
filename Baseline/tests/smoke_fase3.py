"""Smoke test — Fase 3 (NAFNet + weak PatchGAN + MIND refinement).

Fabricates tiny data, launches a real (2-3 step) refinement with train_fase3.py
(from scratch, no warm-start needed for the smoke), then runs inference and
checks the output volume. Also verifies warm-starting from a Fase 2 checkpoint.

Run:
    python tests/smoke_fase3.py
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
    work = Path(tempfile.mkdtemp(prefix="smoke3_"))
    prep = work / "prep"
    U.make_fake_prep(prep, H=64, W=64, n_slices=8)

    cfg = dict(
        task=2, task_name="smoke_task2g", method="nafnet_gan",
        model=U.base_model_cfg(),
        disc=dict(ndf=8, n_layers=2, gan_mode="lsgan"),
        data=dict(modality="T1W", source_field="0.1T", target_field="3T", n_neighbors=1),
        loss=dict(l1=1.0, ssim=1.0, edge=0.1, lpips=0.0, adv=0.05, mind=0.5, mind_radius=2),
        train=U.base_train_cfg(pretrain_epochs=0, finetune_epochs=1,
                               finetune_lr=5e-5, disc_lr=1e-4),
    )
    cfg_path = U.write_yaml(work / "smoke_fase3.yaml", cfg)

    # --- launch refinement (CPU, 3 steps) ---
    U.run_script("train_fase3.py",
                 ["--config", cfg_path, "--preprocessed_dir", prep,
                  "--bbox", prep / "bbox.json", "--max_steps", "3"], work)

    ckpt = work / "out" / "smoke_task2g" / "nafnet_gan" / "finetune_last.pth"
    assert ckpt.exists(), f"missing checkpoint {ckpt}"
    print(f"[train_fase3] ok -> {ckpt.name}")

    # --- warm-start from that checkpoint (exercises --init_from path) ---
    U.run_script("train_fase3.py",
                 ["--config", cfg_path, "--preprocessed_dir", prep,
                  "--bbox", prep / "bbox.json", "--init_from", ckpt, "--use_ema",
                  "--max_steps", "2"], work)
    print("[train_fase3 --init_from] ok")

    # --- inference (reuse inference_fase2: same generator arch) ---
    in_dir = U.make_fake_nifti_inputs(work / "nifti_in")
    U.run_script("inference_fase2.py",
                 ["--config", cfg_path, "--checkpoint", ckpt,
                  "--input_dir", in_dir, "--output_dir", work / "inf",
                  "--bbox", prep / "bbox.json", "--use_ema", "--tta"], work)

    outs = list((work / "inf").rglob("*.nii.gz"))
    assert outs, "no inference outputs"
    import nibabel as nib
    arr = nib.load(str(outs[0])).get_fdata()
    assert np.isfinite(arr).all()
    print(f"[inference] ok -> {len(outs)} volumes, finite")

    print("\n===== FASE 3 SMOKE PASSED =====")


if __name__ == "__main__":
    main()
