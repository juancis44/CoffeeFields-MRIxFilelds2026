"""Smoke test — Fase 6 (evaluation + ranking vs baseline).

Two parts:
    A) compare_methods ranking on hand-made summaries (deterministic, no deps).
    B) a real Evaluation/evaluate.py run (nRMSE/SSIM) on fake matched volumes,
       then feed its JSON through compare_methods.

Run:
    python tests/smoke_fase6.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent          # Baseline/
CHALLENGE = REPO.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import _smoke_utils as U


def make_matched_volumes(pred_dir, tgt_dir, noise, seed):
    """Create pred/target .nii.gz with matching subject IDs; pred = target+noise."""
    import nibabel as nib
    pred_dir.mkdir(parents=True, exist_ok=True); tgt_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for sid in ("0001", "0002"):
        vol = np.stack([U.brain_slice(48, 48, rng) for _ in range(6)], axis=2)
        nib.save(nib.Nifti1Image(vol.astype(np.float32), np.eye(4)),
                 str(tgt_dir / f"P_T1W_3T_{sid}.nii.gz"))
        pred = np.clip(vol + rng.normal(0, noise, vol.shape), 0, 1).astype(np.float32)
        nib.save(nib.Nifti1Image(pred, np.eye(4)),
                 str(pred_dir / f"P_T1W_3T_{sid}.nii.gz"))


def compare(entries, work):
    cmd = [sys.executable, str(REPO / "scripts" / "compare_methods.py"), *entries]
    print(f"\n$ {' '.join(cmd)}")
    r = subprocess.run(cmd)
    assert r.returncode == 0


def main():
    work = Path(tempfile.mkdtemp(prefix="smoke6_"))

    # --- A) ranking logic on synthetic summaries ---
    base = {"nrmse_mean": 0.20, "ssim_mean": 0.80, "lpips_mean": 0.30, "dice_mean": 0.70, "volume_mean": 0.85}
    good = {"nrmse_mean": 0.12, "ssim_mean": 0.90, "lpips_mean": 0.18, "dice_mean": 0.80, "volume_mean": 0.92}
    (work / "baseline.json").write_text(json.dumps(base))
    (work / "nafnet.json").write_text(json.dumps(good))
    # import ranking directly to assert winner
    sys.path.insert(0, str(REPO / "scripts"))
    import compare_methods as C
    ranks, totals = C.rank_methods(
        {"baseline": {k[:-5]: v for k, v in base.items()},
         "nafnet": {k[:-5]: v for k, v in good.items()}},
        ["nrmse", "ssim", "lpips", "dice", "volume"])
    assert totals["nafnet"] < totals["baseline"], totals
    assert totals["nafnet"] == 5.0 and totals["baseline"] == 10.0, totals  # 1s vs 2s over 5 metrics
    print(f"[ranking] ok  totals={totals} (nafnet wins 5 vs 10)")
    compare([f"baseline={work/'baseline.json'}", f"nafnet={work/'nafnet.json'}"], work)

    # --- B) real evaluate.py (nrmse/ssim) then compare ---
    pred = work / "pred"; tgt = work / "tgt"
    make_matched_volumes(pred, tgt, noise=0.05, seed=1)
    out_json = work / "results" / "nafnet_eval.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, str(CHALLENGE / "Evaluation" / "evaluate.py"),
                        "--pred_dir", str(pred), "--target_dir", str(tgt),
                        "--metrics", "nrmse", "ssim", "--device", "cpu",
                        "--output_json", str(out_json)])
    assert r.returncode == 0 and out_json.exists(), "evaluate.py failed"
    summ = json.loads(out_json.read_text())
    assert "nrmse_mean" in summ and "ssim_mean" in summ, summ
    print(f"[evaluate.py] ok  nRMSE={summ['nrmse_mean']:.4f} SSIM={summ['ssim_mean']:.4f}")

    print("\n===== FASE 6 SMOKE PASSED =====")


if __name__ == "__main__":
    main()
