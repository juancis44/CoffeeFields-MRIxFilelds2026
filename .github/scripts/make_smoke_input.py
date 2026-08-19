#!/usr/bin/env python3
"""Build a tiny synthetic input + manifest.json for the Docker smoke test.

This is NOT a correctness/accuracy test -- it is a "does the container run
end-to-end without crashing" test: build the image, feed it a fake volume,
confirm it picks the right checkpoint, runs the crop/predict/paste/orientation
pipeline, and writes a valid output volume in range.

Kept deliberately small (4 axial slices) so it finishes in CI on CPU in a
reasonable time -- inference.py still processes each slice at the full
production crop size (falls back to the 364x436 default bbox when no
bbox.json was baked into the image), so slice count is what controls runtime,
not in-plane size.
"""
import json
import os
import sys

import numpy as np
import nibabel as nib

ROOT = sys.argv[1] if len(sys.argv) > 1 else "smoke_input"
DEPTH = 4  # number of axial slices; keep small for CI speed

os.makedirs(ROOT, exist_ok=True)

# Fake brain-ish volume: nonzero in the middle, zero border (so the
# background-masking step in inference.py has something to mask).
vol = np.zeros((48, 48, DEPTH), dtype=np.float32)
vol[8:40, 8:40, :] = np.random.default_rng(0).uniform(0.1, 0.6, size=(32, 32, DEPTH)).astype(np.float32)

affine = np.eye(4, dtype=np.float32)
img = nib.Nifti1Image(vol, affine)

input_rel = "T1W/0.1T_to_3T/src/0001.nii.gz"
output_rel = "T1W/0.1T_to_3T/pred/0001.nii.gz"

input_path = os.path.join(ROOT, input_rel)
os.makedirs(os.path.dirname(input_path), exist_ok=True)
nib.save(img, input_path)

manifest = {
    "task": "task2-smoke-test",
    "samples": [
        {
            "source_field": "0.1T",
            "target_field": "3T",
            "modality": "T1W",
            "input": input_rel,
            "output": output_rel,
        }
    ],
}
with open(os.path.join(ROOT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Smoke input written under {ROOT}/ ({DEPTH} slices, expects checkpoint "
      f"task2_0.1T_to_3T_T1W_da_semi_last.pth)")
print(f"Expected output at: {output_rel}")
