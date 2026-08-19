#!/usr/bin/env python3
"""Validate the smoke test's output volume: exists, right dtype, in [0, 1]."""
import sys

import nibabel as nib
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "smoke_output/T1W/0.1T_to_3T/pred/0001.nii.gz"

img = nib.load(path)
data = img.get_fdata(dtype=np.float32)

print(f"Loaded {path}")
print(f"  shape: {data.shape}")
print(f"  dtype: {data.dtype}")
print(f"  range: [{data.min():.4f}, {data.max():.4f}]")

assert data.min() >= -1e-4, f"output below 0: min={data.min()}"
assert data.max() <= 1.0 + 1e-4, f"output above 1: max={data.max()}"
assert np.isfinite(data).all(), "output contains NaN/inf"

print("OK: smoke test output is well-formed.")
