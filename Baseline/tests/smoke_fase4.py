"""Smoke test — Fase 4 (semi-supervised: paired + unpaired retrospective).

Fabricates tiny data (incl. unpaired retro 0.1T) and launches a real 2-3 step
semi-supervised training with train_fase4.py, then runs inference.

Run:
    python tests/smoke_fase4.py
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
    work = Path(tempfile.mkdtemp(prefix="smoke4_"))
    prep = work / "prep"
    U.make_fake_prep(prep, H=64, W=64, n_slices=8)   # now also writes retro 0.1T (unpaired)

    cfg = dict(
        task=2, task_name="smoke_task2s", method="nafnet_semi",
        model=U.base_model_cfg(),
        disc=dict(ndf=8, n_layers=2, gan_mode="lsgan"),
        data=dict(modality="T1W", source_field="0.1T", target_field="3T", n_neighbors=1),
        loss=dict(l1=1.0, ssim=1.0, edge=0.1, lpips=0.0, adv=0.05, mind=0.5,
                  mind_radius=2, adv_unpaired=0.05, mind_unpaired=0.5),
        train=U.base_train_cfg(finetune_epochs=1, finetune_lr=5e-5, disc_lr=1e-4),
    )
    cfg_path = U.write_yaml(work / "smoke_fase4.yaml", cfg)

    U.run_script("train_fase4.py",
                 ["--config", cfg_path, "--preprocessed_dir", prep,
                  "--bbox", prep / "bbox.json", "--max_steps", "3"], work)

    ckpt = work / "out" / "smoke_task2s" / "nafnet_semi" / "semi_last.pth"
    assert ckpt.exists(), f"missing checkpoint {ckpt}"
    print(f"[train_fase4] ok -> {ckpt.name}")

    in_dir = U.make_fake_nifti_inputs(work / "nifti_in")
    U.run_script("inference_fase2.py",
                 ["--config", cfg_path, "--checkpoint", ckpt,
                  "--input_dir", in_dir, "--output_dir", work / "inf",
                  "--bbox", prep / "bbox.json", "--use_ema"], work)
    outs = list((work / "inf").rglob("*.nii.gz"))
    assert outs, "no inference outputs"
    import nibabel as nib
    assert np.isfinite(nib.load(str(outs[0])).get_fdata()).all()
    print(f"[inference] ok -> {len(outs)} volumes, finite")

    print("\n===== FASE 4 SMOKE PASSED =====")


if __name__ == "__main__":
    main()
