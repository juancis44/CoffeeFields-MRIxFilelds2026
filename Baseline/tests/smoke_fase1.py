"""Smoke test — Fase 1 (data pipeline).

Fabricates a tiny fake dataset and exercises every Fase 1 component:
    degrade -> compute_bbox script -> RAMCachedPaired25D / SyntheticLowfieldPaired25D
    (through a real DataLoader) -> qc_registration script.

Run:
    python tests/smoke_fase1.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import _smoke_utils as U
from torch.utils.data import DataLoader

from mrixfields.data.degrade import simulate_lowfield, DegradeConfig
from mrixfields.data.brain_bbox import load_bbox
from mrixfields.data.paired25d import RAMCachedPaired25D, SyntheticLowfieldPaired25D


def main():
    work = Path(tempfile.mkdtemp(prefix="smoke1_"))
    prep = work / "prep"
    U.make_fake_prep(prep, H=64, W=64, n_slices=8)
    print(f"fake prep -> {prep}")

    # 1) degrade
    clean = U.brain_slice(64, 64, np.random.default_rng(0))
    deg = simulate_lowfield(clean, DegradeConfig(), seed=1)
    assert deg.shape == (64, 64) and 0 <= deg.min() and deg.max() <= 1.0 + 1e-5
    print("[degrade] ok")

    # 2) compute_bbox script (overwrites bbox.json)
    U.run_script("compute_bbox.py",
                 ["--preprocessed_dir", prep, "--split", "pro_train",
                  "--modality", "T1W", "--field", "3T", "--snap", "8",
                  "--out", prep / "bbox.json"], work)
    bbox = load_bbox(prep / "bbox.json")
    print(f"[compute_bbox] ok -> {bbox['row']} {bbox['col']}")

    # 3) real paired dataset through a DataLoader (K=1 -> 3 input channels)
    ds = RAMCachedPaired25D(prep, "pro_train", "T1W", "0.1T", "3T",
                            bbox=bbox, n_neighbors=1, augment=True, cache=True)
    dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)
    b = next(iter(dl))
    Hc, Wc = bbox["row"][1] - bbox["row"][0], bbox["col"][1] - bbox["col"][0]
    assert tuple(b["source"].shape) == (2, 3, Hc, Wc), b["source"].shape
    assert tuple(b["target"].shape) == (2, 1, Hc, Wc), b["target"].shape
    assert float(b["source"].min()) >= -1.0001 and float(b["source"].max()) <= 1.0001
    print(f"[paired25d+DataLoader] ok  src={tuple(b['source'].shape)} tgt={tuple(b['target'].shape)} range[-1,1]")

    # 4) synthetic pretrain dataset
    sds = SyntheticLowfieldPaired25D(prep, "retro_train", "T1W", "3T",
                                     bbox=bbox, n_neighbors=1, augment=True, cache=True)
    sb = next(iter(DataLoader(sds, batch_size=2, num_workers=0)))
    assert tuple(sb["source"].shape) == (2, 3, Hc, Wc)
    print(f"[synthetic+DataLoader] ok  n={len(sds)}")

    # 5) qc_registration script on paired NIfTI
    data_dir = work / "data"
    U.make_fake_nifti_pairs(data_dir)
    U.run_script("qc_registration.py",
                 ["--data_dir", data_dir, "--split", "Training_prospective",
                  "--modality", "T1W", "--source", "0.1T", "--target", "3T",
                  "--out", work / "qc.csv"], work)
    assert (work / "qc.csv").exists()
    print("[qc_registration] ok")

    print("\n===== FASE 1 SMOKE PASSED =====")


if __name__ == "__main__":
    main()
