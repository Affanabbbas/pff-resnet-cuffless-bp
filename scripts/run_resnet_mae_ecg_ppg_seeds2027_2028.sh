#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
DATA="${MIMIC_BP_DIR:-../../datasets/mimic_bp}"

mkdir -p logs outputs checkpoints

SUMMARY_CSV="outputs/resnet_mae_ecg_ppg_seeds2026_2028_summary.csv"

fail_if_exists() {
  local path="$1"
  if [ -e "$path" ]; then
    echo "Refusing to overwrite existing file/path: $path" >&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  if [ ! -f "$path" ]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
}

echo "Validating ECG+PPG seed configs and command inputs..."

python - <<'PY'
from pathlib import Path
import sys
import yaml

base_path = Path("config_resnet_mae_ecg_ppg_seed2026.yaml")
cfg_paths = {
    2026: base_path,
    2027: Path("config_resnet_mae_ecg_ppg_seed2027.yaml"),
    2028: Path("config_resnet_mae_ecg_ppg_seed2028.yaml"),
}

base = yaml.safe_load(base_path.read_text())
errors = []

for seed, path in cfg_paths.items():
    if not path.exists():
        errors.append(f"missing {path}")
        continue

    cfg = yaml.safe_load(path.read_text())
    expected_save_dir = f"checkpoints/resnet_mae_ecg_ppg_seed{seed}"
    expected_log_csv = f"outputs/resnet_mae_ecg_ppg_seed{seed}_train_log.csv"

    if cfg.get("training", {}).get("seed") != seed:
        errors.append(f"{path}: training.seed is not {seed}")
    if cfg.get("training", {}).get("save_dir") != expected_save_dir:
        errors.append(f"{path}: training.save_dir is not {expected_save_dir}")
    if cfg.get("training", {}).get("log_csv") != expected_log_csv:
        errors.append(f"{path}: training.log_csv is not {expected_log_csv}")

    comparable = yaml.safe_load(path.read_text())
    comparable["training"]["seed"] = 2026
    comparable["training"]["save_dir"] = base["training"]["save_dir"]
    comparable["training"]["log_csv"] = base["training"]["log_csv"]
    if comparable != base:
        errors.append(f"{path}: differs from seed-2026 config beyond seed/save_dir/log_csv")

if errors:
    print("Config validation failed:", file=sys.stderr)
    for err in errors:
        print(f"  - {err}", file=sys.stderr)
    sys.exit(1)

print("Config validation passed.")
PY

require_file "outputs/resnet_mae_ecg_ppg_seed2026_test_metrics.json"
require_file "outputs/resnet_mae_ecg_ppg_seed2026_test_predictions.csv"
require_file "outputs/resnet_mae_ecg_ppg_seed2026_test_subject_predictions.csv"

python -m py_compile train.py evaluate.py
python train.py --help >/dev/null
python evaluate.py --help >/dev/null

if [ -e "$SUMMARY_CSV" ]; then
  echo "Summary already exists; it will be left unchanged until all requested metrics are present: $SUMMARY_CSV"
fi

for SEED in 2027 2028
do
  CONFIG="config_resnet_mae_ecg_ppg_seed${SEED}.yaml"
  CKPT_DIR="checkpoints/resnet_mae_ecg_ppg_seed${SEED}"
  CKPT="${CKPT_DIR}/best_resnet1d.pt"
  TRAIN_LOG="outputs/resnet_mae_ecg_ppg_seed${SEED}_train_log.csv"
  TEST_PRED="outputs/resnet_mae_ecg_ppg_seed${SEED}_test_predictions.csv"
  TEST_SUBJECT="outputs/resnet_mae_ecg_ppg_seed${SEED}_test_subject_predictions.csv"
  TEST_METRICS="outputs/resnet_mae_ecg_ppg_seed${SEED}_test_metrics.json"
  TRAIN_TERMINAL_LOG="logs/resnet_mae_ecg_ppg_seed${SEED}_train_terminal.log"
  EVAL_TERMINAL_LOG="logs/resnet_mae_ecg_ppg_seed${SEED}_eval_terminal.log"

  echo
  echo "======================================================"
  echo "ECG+PPG ResNet1D MAE seed ${SEED}"
  echo "======================================================"

  if [ -f "$TEST_METRICS" ]; then
    echo "Seed ${SEED} already has test metrics. Skipping training/evaluation."
    continue
  fi

  fail_if_exists "$CKPT_DIR"
  fail_if_exists "$TRAIN_LOG"
  fail_if_exists "$TEST_PRED"
  fail_if_exists "$TEST_SUBJECT"
  fail_if_exists "$TRAIN_TERMINAL_LOG"
  fail_if_exists "$EVAL_TERMINAL_LOG"

  echo "Training seed ${SEED}..."
  python train.py --config "$CONFIG" 2>&1 | tee "$TRAIN_TERMINAL_LOG"

  require_file "$CKPT"

  echo "Evaluating seed ${SEED}..."
  python evaluate.py \
    --checkpoint "$CKPT" \
    --data_dir "$DATA" \
    --split test \
    --batch_size 64 \
    --num_workers 4 \
    --out_csv "$TEST_PRED" \
    --subject_csv "$TEST_SUBJECT" \
    --metrics_json "$TEST_METRICS" \
    2>&1 | tee "$EVAL_TERMINAL_LOG"
done

if [ -e "$SUMMARY_CSV" ]; then
  echo "Refusing to overwrite existing summary: $SUMMARY_CSV" >&2
  exit 1
fi

echo
echo "Writing ECG+PPG 3-seed summary CSV..."

python - <<'PY'
import csv
import json
from pathlib import Path

seeds = [2026, 2027, 2028]
summary_path = Path("outputs/resnet_mae_ecg_ppg_seeds2026_2028_summary.csv")

if summary_path.exists():
    raise FileExistsError(f"Refusing to overwrite {summary_path}")

rows = []
metrics = [
    "SBP_MAE",
    "DBP_MAE",
    "AVG_MAE",
    "SBP_RMSE",
    "DBP_RMSE",
    "AVG_RMSE",
    "SBP_Pearson",
    "DBP_Pearson",
]
subject_metrics = [
    "SBP_MAE",
    "DBP_MAE",
    "AVG_MAE",
    "SBP_Pearson",
    "DBP_Pearson",
]

for seed in seeds:
    path = Path(f"outputs/resnet_mae_ecg_ppg_seed{seed}_test_metrics.json")
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics for seed {seed}: {path}")
    data = json.loads(path.read_text())
    row = {"row_type": "seed", "model": "ResNet1D_ECG_PPG_MAE", "seed": seed}
    for key in metrics:
        row[key] = data[key]
    for key in subject_metrics:
        row[f"subject_{key}"] = data["subject_level"][key]
    rows.append(row)

fieldnames = list(rows[0].keys())
numeric_fields = [field for field in fieldnames if field not in {"row_type", "model", "seed"}]

for stat in ["mean", "std", "min", "max"]:
    row = {"row_type": stat, "model": "ResNet1D_ECG_PPG_MAE", "seed": ""}
    for field in numeric_fields:
        values = [float(seed_row[field]) for seed_row in rows]
        if stat == "mean":
            value = sum(values) / len(values)
        elif stat == "std":
            mean = sum(values) / len(values)
            value = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
        elif stat == "min":
            value = min(values)
        else:
            value = max(values)
        row[field] = value
    rows.append(row)

summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {summary_path}")
PY
