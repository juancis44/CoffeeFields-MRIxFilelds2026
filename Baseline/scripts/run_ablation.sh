#!/usr/bin/env bash
# Ablation for the MICCAI reviewer response: isolate MIND's contribution to Stage B.
#
# Both reviewers flagged that Stage B changes 4 things at once relative to Stage A
# (real 0.1T data, MIND loss, more epochs, lower LR), so the observed gain can't be
# attributed to MIND specifically. This produces the two controls they asked for,
# holding epoch count and LR fixed at Stage B's values in both:
#
#   Stage A              = existing pretrain checkpoint (no run needed)
#   Control 1 (no-real)  = continue SYNTHETIC-only training, same epochs/LR as Stage B,
#                          no real 0.1T exposure, no MIND -> isolates "more training"
#   Control 2 (no-MIND)  = Stage B's real-data recipe verbatim but loss.mind=0 and
#                          loss.mind_unpaired=0 -> isolates MIND's own contribution
#   Stage B (existing)   = the model you already trained (real data + MIND)
#
# Usage:
#   ./run_ablation.sh T1W 3T
#
# IMPORTANT: point this at a combo whose Stage A/B checkpoints were trained with the
# CORRECT settings (--pretrain_epochs 80 --da_epochs 25 --max_train_subjects 0).
# Run the config audit first (see chat) -- some early combos in this sweep silently
# reused stale configs from the earlier calibration run (65-subject cap, 30 epochs).
# Using one of those as "Stage A/B" here would invalidate the whole ablation.
set -euo pipefail

MOD="${1:?usage: run_ablation.sh MODALITY TARGET_FIELD   e.g. run_ablation.sh T1W 3T}"
FIELD="${2:?usage: run_ablation.sh MODALITY TARGET_FIELD   e.g. run_ablation.sh T1W 3T}"
COMBO="0.1T_to_${FIELD}_${MOD}"

ROOT=/home/juanc/challenge/Baseline
PREP=/data/juan/preprocessed
BBOX="$PREP/bbox.json"
RUNS=/data/juan/runs
LOGDIR="$ROOT/logs/ablation"
mkdir -p "$LOGDIR"

STAGE_A_CKPT="$RUNS/task2_${COMBO}_interim/nafnet/pretrain_last.pth"
DA_CFG="$ROOT/configs/task2/sweep/da/${COMBO}.yaml"
BASE_INTERIM_CFG="$ROOT/configs/task2/sweep/interim/${COMBO}.yaml"

if [ ! -f "$STAGE_A_CKPT" ]; then
  echo "ERROR: Stage A checkpoint not found: $STAGE_A_CKPT" >&2
  exit 1
fi
if [ ! -f "$DA_CFG" ]; then
  echo "ERROR: DA config not found: $DA_CFG" >&2
  exit 1
fi

echo "=== Ablation for $COMBO ==="
echo "Stage A checkpoint: $STAGE_A_CKPT"

# Sanity print: show the epochs/subjects this combo actually used, so you can eyeball
# whether it's the full-settings run or a stale one before burning GPU time.
python3 - "$DA_CFG" "$BASE_INTERIM_CFG" <<'PY'
import sys, yaml
da = yaml.safe_load(open(sys.argv[1]))
pre = yaml.safe_load(open(sys.argv[2]))
print(f"  pretrain: epochs={pre['train'].get('pretrain_epochs')} "
      f"max_subjects={pre['data'].get('max_train_subjects', pre['data'].get('max_subjects'))}")
print(f"  DA:       epochs={da['train'].get('finetune_epochs')} lr={da['train'].get('finetune_lr')} "
      f"mind={da['loss'].get('mind')} mind_unpaired={da['loss'].get('mind_unpaired')}")
PY
read -p "Settings look right for this combo? [y/N] " ok
[ "$ok" = "y" ] || { echo "Aborting -- fix the config or pick a different combo."; exit 1; }

# ---- Control 1: same-length continuation, synthetic only, no real data, no MIND ----
C1_CFG="$ROOT/configs/task2/sweep/interim/${COMBO}_control1.yaml"
python3 - "$DA_CFG" "$BASE_INTERIM_CFG" "$C1_CFG" <<'PY'
import sys, yaml
da_path, base_path, out_path = sys.argv[1:4]
da = yaml.safe_load(open(da_path))
epochs = da["train"]["finetune_epochs"]
lr = da["train"]["finetune_lr"]
cfg = yaml.safe_load(open(base_path))
cfg["train"]["pretrain_epochs"] = epochs
cfg["train"]["pretrain_lr"] = lr
cfg["task_name"] = cfg.get("task_name", "") + "_ablation_control1_synthonly"
yaml.safe_dump(cfg, open(out_path, "w"))
print(f"[control1] wrote {out_path}: pretrain_epochs={epochs} pretrain_lr={lr}")
PY

torchrun --nproc_per_node=2 "$ROOT/scripts/train_fase2.py" \
  --config "$C1_CFG" \
  --preprocessed_dir "$PREP" --bbox "$BBOX" \
  --init_from "$STAGE_A_CKPT" \
  2>&1 | tee "$LOGDIR/${COMBO}_control1.log"

# ---- Control 2: Stage B's real-data recipe verbatim, MIND OFF ----
C2_CFG="$ROOT/configs/task2/sweep/da/${COMBO}_control2_noMIND.yaml"
python3 - "$DA_CFG" "$C2_CFG" <<'PY'
import sys, yaml
src, dst = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(src))
cfg["loss"]["mind"] = 0.0
cfg["loss"]["mind_unpaired"] = 0.0
cfg["task_name"] = cfg.get("task_name", "") + "_ablation_control2_noMIND"
yaml.safe_dump(cfg, open(dst, "w"))
print(f"[control2] wrote {dst}: mind=0.0 mind_unpaired=0.0 (everything else unchanged)")
PY

torchrun --nproc_per_node=2 "$ROOT/scripts/train_fase4.py" \
  --config "$C2_CFG" \
  --preprocessed_dir "$PREP" --bbox "$BBOX" \
  --init_from "$STAGE_A_CKPT" --use_ema \
  2>&1 | tee "$LOGDIR/${COMBO}_control2.log"

echo ""
echo "=== Done: Stage A, Control1, Control2 checkpoints ready (Stage B = your existing DA model). ==="
echo "Next steps (do these on the OFFICIAL validation split, not synthetic held-out):"
echo "  1) run_submission.py / inference for each of the 4 checkpoints"
echo "  2) Evaluation/evaluate.py  -> nRMSE / SSIM / LPIPS per checkpoint"
echo "  3) segment_predictions.py + check_dgm_labels.py -> DGM Dice/Volume per checkpoint"
echo "  4) tabulate: Stage A | Control1 (no real, no MIND) | Control2 (real, no MIND) | Stage B (real+MIND)"
