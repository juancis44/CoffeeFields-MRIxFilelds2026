# MRIxFields2026 — Task 2: Ultra-Low-Field MRI Enhancement

This repository contains our submission to **Task 2** of the [MRIxFields2026 challenge](https://mrixfields.chihucloud.com/2026/) (MICCAI 2026): translating ultra-low-field (0.1~T) brain MRI into higher-field-equivalent images (1.5~T, 3~T, 5~T, 7~T) across three contrasts (T1W, T2W, T2FLAIR), without access to any real paired training data at the target fields.

The full method, all reported numbers, and an honest account of what worked and what did not are written up in [`PAPER_Task2.tex`](PAPER_Task2.tex). This README summarizes the approach and points to the code that produced each result; it does not reproduce the challenge's own participant guide, which lives in the [official repository](https://github.com/MRIxFields/MRIxFields2026).

## The Challenge, Briefly

MRI hardware spans a wide range of field strengths, from portable ultra-low-field (0.1~T) scanners with poor signal-to-noise ratio to ultra-high-field (7~T) research systems with exceptional detail but limited availability. MRIxFields2026 Task 2 asks: given a 0.1~T scan, can a model recover the tissue contrast and structural information a higher-field scan would show? The challenge scores submissions on five metrics (nRMSE, SSIM, LPIPS, and Dice/volume-consistency on 14 deep-gray-matter structures via SynthSeg segmentation) and, critically, its released validation split turned out not to contain usable paired ground truth for any field/contrast combination — a data property that shaped most of our design decisions (see below).

## What We Did

- **Backbone**: a 2.5D NAFNet restoration network (five-channel input: the central slice plus two neighbors on each side), chosen for being competitive with transformer-based restoration models at a fraction of the compute cost.
- **Two-stage training**: Stage A pretrains on synthetically degraded high-field volumes (correct pairs by construction, but an easier problem than the real task, since relaxometric contrast change is not modeled by degradation alone). Stage B adapts to real 0.1~T input using an unpaired structural loss (MIND) against real high-field anatomy, since no real paired data exists to supervise this stage directly.
- **A measured, not assumed, degradation simulator**: we audited the synthetic 0.1~T generator with a domain classifier (AUC of "real vs. synthetic" separability) instead of trusting it by construction, found and fixed a background-noise leak and an interpolation spectral fingerprint, and added a contrast quantile map after discovering the simulator never modeled the relaxometric contrast shift between fields at all. The audit is not fully closed — see the paper's Limitations.
- **Ground-truth-free validation**: because the released validation split has no usable pairing (confirmed exhaustively across all twelve target-field/contrast combinations), we built two proxies instead of relying on paired metrics: presence/plausibility of the 14 scored deep-gray-matter structures via SynthSeg voxel counts, and physical-volume (mm³) reference ranges from real target-field anatomy.
- **An ablation isolating MIND's actual contribution** (Stage A vs. two controls vs. full Stage B), which turned out to be one of the least expected findings in the project: most of the measured gain came from additional training time, not from the unpaired structural-adaptation mechanism itself.
- **An honest account of a disagreement between our proxies and the real leaderboard**: every local validation signal indicated the revised pipeline should improve on the original submission; the actual leaderboard result regressed slightly (primary SSIM 0.7994 vs. 0.8060). We ruled out one plausible mechanism (k-space-truncation ringing) but do not have a confirmed explanation, and we say so directly rather than paper over it.
- **A specific, well-characterized failure mode**: T2FLAIR structure synthesis is measurably worse than T1W/T2W (reduced tissue contrast, missing DGM structures under segmentation) and does not respond to post-hoc contrast correction — we treat this as an open problem, not a solved one.

## Repository Structure

```
.
├── PAPER_Task2.tex          # Full write-up: method, all results, limitations
├── references.bib
├── mosaic_physical_all.png  # Qualitative QC figure referenced in the paper
├── Baseline/
│   ├── mrixfields/          # Model, losses, data pipeline (installable package)
│   ├── scripts/             # Final pipeline: preprocessing -> training -> inference -> submission,
│   │                         #  plus the scripts behind every table/figure in the paper
│   ├── configs/task2/       # Training configs actually used for Task 2
│   └── tests/                # Per-phase smoke tests
├── Evaluation/               # Challenge's official metric computation (Dice/Volume/nRMSE/SSIM/LPIPS)
└── Submission/                # Submission packaging + the official scorer
```

This is a **minimal, results-oriented** version of the codebase: it includes the code needed to reproduce the final pipeline and every number/figure reported in the paper, not the full history of exploratory scripts, ablations-of-ablations, and dead ends that got us there.

### Key scripts, mapped to what they produced

| Script | What it's for |
|---|---|
| `Baseline/scripts/preprocess.py`, `compute_bbox.py` | Data preparation (slice extraction, shared brain bounding box) |
| `Baseline/scripts/train_fase2.py` | Stage A: synthetic pretraining |
| `Baseline/scripts/train_fase4.py` | Stage B: unpaired domain adaptation (MIND) |
| `Baseline/scripts/sweep_task2.py` | Propagates the validated recipe across all 12 field × modality combinations |
| `Baseline/scripts/inference_fase5.py` | Final inference (EMA weights, TTA, histogram matching) |
| `Baseline/scripts/run_submission.py`, `prepare_submission_tree.py`, `segment_predictions.py` | Builds predictions + segmentations into the challenge's expected layout |
| `Baseline/scripts/run_ablation.sh` | Stage A / Control 1 / Control 2 / Stage B ablation (Table in §5.4 of the paper) |
| `Baseline/scripts/check_dgm_labels.py`, `reference_dgm_ranges.py` | Ground-truth-free validation proxies (§3.1.1, §5.1) |
| `Baseline/scripts/mosaic_inference.py` | The qualitative QC figure in the paper |

## Quick Start

```bash
conda env create -f environment.yml
conda activate mf
cp .env.example .env   # edit DATA_DIR, PREPROCESSED_DIR, OUTPUT_DIR, etc.
```

From there: `preprocess.py` → `train_fase2.py` (Stage A) → `train_fase4.py` (Stage B) → `inference_fase5.py` → `segment_predictions.py` → `prepare_submission_tree.py`. See `Baseline/README.md` for exact flags and `Evaluation/README.md` for setting up SynthSeg. The full compute budget, library versions, and hardware are listed in the paper's Reproducibility section.

## Results Summary

| Metric (primary, adjusted mean SSIM) | Value |
|---|---|
| Original submission | 0.8060 |
| Revised pipeline (this repo) | 0.7994 |

Per-contrast Dice/volume-consistency, the ablation table, the simulator audit, and the full discussion of the SSIM regression are in the paper — this table is a pointer, not a substitute for reading it.

## License

Code: MIT License ([LICENSE](LICENSE)). The dataset and the official challenge infrastructure this repository builds on belong to the MRIxFields2026 organizers; see the [official repository](https://github.com/MRIxFields/MRIxFields2026) for the dataset, participant guide, and challenge rules.

## Contact

Juan Cisneros — cisneros.juan93@gmail.com
