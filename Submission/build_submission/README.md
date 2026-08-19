# build_submission

Pack Baseline inference outputs into the Synapse submission tree:
copy + rename + axial slice + reorganize, ready for `zip -r task{N}.zip ...`.

## What this tool does

Inference produces `{INFERENCE_DIR}/{task}_{pair}_{MOD}/{method}/{mode}/{epoch}/...` with **source** field tags. The Synapse evaluator wants a different layout (see [Submission/README.md](../README.md)). Four differences:

| Aspect | Inference output | Submission format |
|---|---|---|
| Top-level dir | `task1_0.1T_to_7T_T1W/` flat | `task1/T1W/0.1T_to_7T/` nested |
| pred / seg | `epoch100/`, `epoch100_seg/` | `pred/`, `seg/` |
| method / mode / epoch | many coexist | exactly one per (modality, pair) |
| Field tag in filename | source (`P_T1W_0.1T_0001.nii.gz`) | target (`P_T1W_7T_0001.nii.gz`) |

This tool fixes all four, and axially slices both pred and seg to `(364, 436, 30)`. Zipping is left to you — see "Self-check" at the bottom.

## Usage

```bash
# Default: read $INFERENCE_DIR, write $SUBMISSION_DIR, all three tasks
python Submission/build_submission/build_submission.py

# Specific task only
python Submission/build_submission/build_submission.py --tasks task1

# Dry-run (print src → dst mapping, copy nothing)
python Submission/build_submission/build_submission.py --dry-run

# Clean each target task/ subdir before building
python Submission/build_submission/build_submission.py --clean

# Override which sweep (method / mode / epoch) is packed for a task
python Submission/build_submission/build_submission.py \
    --task1-method cyclegan --task1-mode pro_scratch \
    --task3-epoch step200000
```

After the script finishes, run the `zip` commands it prints (one per task):

```bash
cd $SUBMISSION_DIR && zip -r ~/task1.zip task1/   # zip root MUST be task1/
cd $SUBMISSION_DIR && zip -r ~/task2.zip task2/   # zip root MUST be task2/
cd $SUBMISSION_DIR && zip -r ~/task3.zip task3/   # zip root MUST be task3/
```

Then upload `~/task{1,2,3}.zip` to [Synapse](https://www.synapse.org/Synapse:syn72060672).

## CLI arguments

```
--predictions-dir PATH    Source directory     (default: $INFERENCE_DIR)
--predictions-seg-dir PATH  Seg source dir     (default: $PREDICTIONS_SEG_DIR)
--output-dir PATH         Target directory     (default: $SUBMISSION_DIR)
--tasks LIST              Comma-separated subset (default: task1,task2,task3)
--task{1,2,3}-method STR  Per-task method override
--task{1,2,3}-mode STR    Per-task mode override
--task{1,2,3}-epoch STR   Per-task epoch_tag override
--dry-run                 Print src → dst mapping, copy nothing
--clean                   `rm -rf` target task/ dirs before building
```

## Defaults

Paths default to `.env` values: `--predictions-dir` to `$INFERENCE_DIR`,
`--predictions-seg-dir` to `$PREDICTIONS_SEG_DIR` (Task 1/2 only),
`--output-dir` to `$SUBMISSION_DIR`.

Per-task sweep selection (which `(method, mode, epoch_tag)` triple gets packed when the flags above aren't set):

| Task | method | mode | epoch_tag |
|---|---|---|---|
| task1 | cut | pro_pretrained | epoch100 |
| task2 | cut | pro_pretrained | epoch100 |
| task3 | stargan_v2 | pro_pretrained | epoch50 |

## Output layout

```
$SUBMISSION_DIR/
├── task1/{T1W,T2W,T2FLAIR}/{4 pairs}/{pred,seg}/  P_{MOD}_{TGT}_{ID}.nii.gz   # 36 + 36
├── task2/{T1W,T2W,T2FLAIR}/{4 pairs}/{pred,seg}/  P_{MOD}_{TGT}_{ID}.nii.gz   # 36 + 36
└── task3/{T1W,T2W,T2FLAIR}/{20 pairs}/pred/       P_{MOD}_{TGT}_{ID}.nii.gz   # 180 (no seg)
```

Total: 324 NIfTI files. Every file has shape `(364, 436, 30)` (axial slab `[150, 180)` of the 0.5 mm dataset grid). Task 3 has no `seg/`.

## Naming rule

Keep the source subject ID; replace the field tag with the target field.
Example: source `P_T1W_0.1T_0001.nii.gz` → `P_T1W_7T_0001.nii.gz` for pair `0.1T_to_7T`.

Subject IDs by source field (copied verbatim, zero-padded):

| Source field | IDs |
|---|---|
| `0.1T_to_*` | 0001, 0002, 0003 |
| `1.5T_to_*` | 0004, 0005, 0008 |
| `3T_to_*`   | 0010, 0011, 0012 |
| `5T_to_*`   | 0013, 0014, 0015 |
| `7T_to_*`   | 0016, 0017, 0018 |

Full naming spec: [Submission/README.md](../README.md#file-format--naming).

## What this tool does NOT do

- **No zip** — `zip -r` is left to you, to avoid clobbering `~/task{N}.zip` accidentally.
- **No segmentation** — for Task 1/2, populate `$PREDICTIONS_SEG_DIR` with [Baseline/scripts/segment_predictions.py](../../Baseline/scripts/segment_predictions.py) (sweep-aware) before running this tool.
- **No metric recompute, no value-range check** — file contents pass through (with axial slicing applied).

## Notes

**Axial slice range.** Both pred and seg are sliced along axial `[150, 180)` via `_copy_with_axial_clip`, producing shape `(364, 436, 30)`. The range is `Z_CLIP_RANGE` from [Baseline/mrixfields/zclip_constants.py](../../Baseline/mrixfields/zclip_constants.py); the GT pack imports the same constant, so pred and GT share the affine origin and the evaluator's shape check passes. To submit at a different range, override the constant — but the GT is fixed at `[150, 180)`, so other ranges will fail evaluation. To submit full volumes, replace `_copy_with_axial_clip` with `shutil.copy2` and override the GT slice the same way; the trade-off is upload size (the slab compresses to roughly 1/10 of a full volume after gzip).

**Seg grid assumption.** Files under `$PREDICTIONS_SEG_DIR/` are expected at 0.5 mm `(364, 436, 364)` — what [Baseline/scripts/segment_predictions.py](../../Baseline/scripts/segment_predictions.py) emits after NN-resampling SynthSeg's native 1 mm output to the prediction grid. The same axial slice then keeps seg and pred on the same `(364, 436, 30)` grid for the evaluator.

**Sweep coordinate match.** Seg files are looked up at `$PREDICTIONS_SEG_DIR/{task}_{pair}_{MOD}/{method}/{mode}/{epoch}/` using the same `(method, mode, epoch)` as pred. Mismatched coordinates produce `missing src seg dir` warnings, and Dice / Volume fall back to worst-case (0.0).

## Self-check (recommended before upload)

With the GT tree in place, run the official scorer locally:

```bash
python Submission/evaluation-2026/score.py \
    -i $SUBMISSION_DIR/task1 -t task1 \
    -g /path/to/Validating_prospective_pack1_ground_truth/task1 \
    -o /tmp/selfcheck1
cat /tmp/selfcheck1/Result/results.json | python -m json.tool | head -20
# Expect: submission_status == "SCORED", Num_Files == "36/36"
```
