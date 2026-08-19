# Final Model Weights

This folder contains the final Stage B checkpoints actually used for the submitted inference (`inference_fase5.py`) — one per (target field, modality) combination, 12 in total:

```
Baseline/weights/
├── task2_0.1T_to_1.5T_T1W_da_semi_last.pth
├── task2_0.1T_to_1.5T_T2W_da_semi_last.pth
├── task2_0.1T_to_1.5T_T2FLAIR_da_semi_last.pth
├── task2_0.1T_to_3T_T1W_da_semi_last.pth
├── task2_0.1T_to_3T_T2W_da_semi_last.pth
├── task2_0.1T_to_3T_T2FLAIR_da_semi_last.pth
├── task2_0.1T_to_5T_T1W_da_semi_last.pth
├── task2_0.1T_to_5T_T2W_da_semi_last.pth
├── task2_0.1T_to_5T_T2FLAIR_da_semi_last.pth
├── task2_0.1T_to_7T_T1W_da_semi_last.pth
├── task2_0.1T_to_7T_T2W_da_semi_last.pth
└── task2_0.1T_to_7T_T2FLAIR_da_semi_last.pth
```

Each is the `semi_last.pth` checkpoint (`nafnet_semi`, Stage B) from `runs/task2_0.1T_to_<TARGET>_<MODALITY>_da/nafnet_semi/semi_last.pth`, renamed to include the run name so it's unambiguous which checkpoint is which. Total size ~371~MB (~31~MB each), under GitHub's 100~MB per-file limit, so plain Git works without needing Git LFS.

Only Stage B ("`_da`") checkpoints are included — Stage A ("`_interim`") checkpoints are the synthetic-pretrain warm-start used internally during training and are not needed for inference.

## Using them

```bash
python Baseline/scripts/inference_fase5.py \
    --checkpoints Baseline/weights/task2_0.1T_to_3T_T1W_da_semi_last.pth \
    ...  # see inference_fase5.py --help for the remaining flags (target field, modality, output dir)
```

Repeat per combination, or loop over all twelve files to reproduce the full submission without retraining.
