"""Smoke test — Fase 5 (test-time ensembling).

Trains a tiny NAFNet (Fase 2), then runs inference_fase5 with flip TTA,
multi-view (axial+coronal+sagittal) and checkpoint averaging (SWA), checking the
output volume is finite.

Run:
    python tests/smoke_fase5.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import _smoke_utils as U


def main():
    work = Path(tempfile.mkdtemp(prefix="smoke5_"))
    prep = work / "prep"
    U.make_fake_prep(prep, H=64, W=64, n_slices=8)

    cfg = dict(
        task=2, task_name="smoke_task2", method="nafnet",
        model=U.base_model_cfg(),
        data=dict(modality="T1W", source_field="0.1T", target_field="3T", n_neighbors=1),
        loss=dict(l1=1.0, ssim=1.0, edge=0.1, lpips=0.0),
        train=U.base_train_cfg(pretrain_epochs=1, pretrain_lr=2e-4,
                               finetune_epochs=1, finetune_lr=1e-4),
    )
    cfg_path = U.write_yaml(work / "smoke_fase5.yaml", cfg)

    U.run_script("train_fase2.py",
                 ["--config", cfg_path, "--preprocessed_dir", prep,
                  "--bbox", prep / "bbox.json", "--max_steps", "3"], work)

    out = work / "out" / "smoke_task2" / "nafnet"
    ck_ft, ck_pt = out / "finetune_last.pth", out / "pretrain_last.pth"
    assert ck_ft.exists() and ck_pt.exists()

    # bbox matched to the fake inference volume (Z=8) so multi-view crops are valid
    bbox_inf = work / "bbox_inf.json"
    bbox_inf.write_text(json.dumps({"row": [0, 64], "col": [0, 64], "slice": [0, 8]}))

    in_dir = U.make_fake_nifti_inputs(work / "nifti_in")   # Z=8 volumes
    U.run_script("inference_fase5.py",
                 ["--config", cfg_path,
                  "--checkpoints", f"{ck_ft},{ck_pt}",       # exercise SWA averaging
                  "--input_dir", in_dir, "--output_dir", work / "inf",
                  "--bbox", bbox_inf, "--tta", "--views", "axial,coronal,sagittal"], work)

    outs = list((work / "inf").rglob("*.nii.gz"))
    assert outs, "no inference outputs"
    import nibabel as nib
    arr = nib.load(str(outs[0])).get_fdata()
    assert np.isfinite(arr).all()
    print(f"[inference_fase5] ok -> {len(outs)} volumes, finite (TTA + 3 views + SWA)")

    print("\n===== FASE 5 SMOKE PASSED =====")


if __name__ == "__main__":
    main()
