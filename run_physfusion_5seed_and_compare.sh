#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
DATA="${MIMIC_BP_DIR:-../../datasets/mimic_bp}"

mkdir -p logs outputs checkpoints

echo "======================================================"
echo "Creating PhysFusion config files for seeds 2026-2030"
echo "======================================================"

python - <<'PY'
import yaml
from pathlib import Path

seeds = [2026, 2027, 2028, 2029, 2030]

base_config = Path("config.yaml")
assert base_config.exists(), "config.yaml not found"

for seed in seeds:
    cfg = yaml.safe_load(base_config.read_text())

    import os
    cfg["data_dir"] = os.environ.get("MIMIC_BP_DIR", "../../datasets/mimic_bp")

    cfg.setdefault("signals", {})
    cfg["signals"]["use_derivatives"] = True
    cfg["signals"]["normalization"] = "per_segment_zscore"
    cfg["signals"]["preprocess"] = True
    cfg["signals"]["quality_filter"] = True
    cfg["signals"]["include_physio_features"] = True

    cfg.setdefault("training", {})
    cfg["training"]["seed"] = seed
    cfg["training"]["model"] = "resnet1d_physfusion"
    cfg["training"]["batch_size"] = 64
    cfg["training"]["epochs"] = 50
    cfg["training"]["lr"] = 0.0001
    cfg["training"]["weight_decay"] = 0.0001
    cfg["training"]["loss"] = "mae"
    cfg["training"]["num_workers"] = 4
    cfg["training"]["label_standardization"] = True
    cfg["training"]["best_metric"] = "AVG_MAE"
    cfg["training"]["save_dir"] = f"checkpoints/physfusion_seed{seed}"
    cfg["training"]["log_csv"] = f"outputs/physfusion_seed{seed}_train_log.csv"

    out = Path(f"config_physfusion_seed{seed}.yaml")
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"✅ Saved {out}")
PY


echo
echo "======================================================"
echo "Running PhysFusion seeds 2026-2030"
echo "Already completed seeds will be skipped"
echo "======================================================"

for SEED in 2026 2027 2028 2029 2030
do
  echo
  echo "======================================"
  echo "PhysFusion seed $SEED"
  echo "======================================"

  if [ -f "outputs/physfusion_seed${SEED}_test_metrics.json" ]; then
    echo "✅ Seed $SEED already completed. Skipping training/evaluation."
    continue
  fi

  echo "Cleaning old partial files for seed $SEED..."
  rm -rf checkpoints/physfusion_seed${SEED}
  rm -f outputs/physfusion_seed${SEED}_*

  echo "Training PhysFusion seed $SEED..."
  python train.py --config config_physfusion_seed${SEED}.yaml \
    2>&1 | tee logs/physfusion_seed${SEED}_train_terminal.log

  CKPT=$(ls checkpoints/physfusion_seed${SEED}/best_*.pt | head -n 1)

  if [ -z "$CKPT" ]; then
    echo "❌ No checkpoint found for seed $SEED"
    exit 1
  fi

  echo "Using checkpoint: $CKPT"

  echo "Evaluating PhysFusion seed $SEED..."
  python evaluate.py \
    --checkpoint "$CKPT" \
    --data_dir "$DATA" \
    --split test \
    --batch_size 64 \
    --num_workers 4 \
    --out_csv outputs/physfusion_seed${SEED}_test_predictions.csv \
    --subject_csv outputs/physfusion_seed${SEED}_test_subject_predictions.csv \
    --metrics_json outputs/physfusion_seed${SEED}_test_metrics.json \
    2>&1 | tee logs/physfusion_seed${SEED}_eval_terminal.log

done


echo
echo "======================================================"
echo "Making PhysFusion 5-seed summary"
echo "======================================================"

python - <<'PY'
import json
import pandas as pd
from pathlib import Path

seeds = [2026, 2027, 2028, 2029, 2030]

rows = []

for seed in seeds:
    path = Path(f"outputs/physfusion_seed{seed}_test_metrics.json")

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    m = json.loads(path.read_text())

    rows.append({
        "model": "PhysFusion",
        "seed": seed,

        "SBP_MAE": m["SBP_MAE"],
        "DBP_MAE": m["DBP_MAE"],
        "AVG_MAE": m["AVG_MAE"],

        "SBP_RMSE": m["SBP_RMSE"],
        "DBP_RMSE": m["DBP_RMSE"],
        "AVG_RMSE": m["AVG_RMSE"],

        "SBP_Pearson": m["SBP_Pearson"],
        "DBP_Pearson": m["DBP_Pearson"],

        "subject_SBP_MAE": m["subject_level"]["SBP_MAE"],
        "subject_DBP_MAE": m["subject_level"]["DBP_MAE"],
        "subject_AVG_MAE": m["subject_level"]["AVG_MAE"],

        "subject_SBP_Pearson": m["subject_level"]["SBP_Pearson"],
        "subject_DBP_Pearson": m["subject_level"]["DBP_Pearson"],
    })

df = pd.DataFrame(rows)

numeric_cols = [
    "SBP_MAE",
    "DBP_MAE",
    "AVG_MAE",
    "SBP_RMSE",
    "DBP_RMSE",
    "AVG_RMSE",
    "SBP_Pearson",
    "DBP_Pearson",
    "subject_SBP_MAE",
    "subject_DBP_MAE",
    "subject_AVG_MAE",
    "subject_SBP_Pearson",
    "subject_DBP_Pearson",
]

summary = df[numeric_cols].agg(["mean", "std", "min", "max"]).T
summary.insert(0, "model", "PhysFusion")

print("\n===== PhysFusion per-seed results =====")
print(df.to_string(index=False))

print("\n===== PhysFusion summary mean ± std =====")
for metric in numeric_cols:
    mean = summary.loc[metric, "mean"]
    std = summary.loc[metric, "std"]
    print(f"{metric}: {mean:.4f} ± {std:.4f}")

df.to_csv("outputs/physfusion_5seed_results.csv", index=False)
summary.to_csv("outputs/physfusion_5seed_summary.csv")

print("\n✅ Saved:")
print("outputs/physfusion_5seed_results.csv")
print("outputs/physfusion_5seed_summary.csv")
PY


echo
echo "======================================================"
echo "Comparing ResNet vs PhysFusion"
echo "======================================================"

python - <<'PY'
import pandas as pd
from pathlib import Path

resnet_path = Path("outputs/resnet_mae_5seed_results.csv")
phys_path = Path("outputs/physfusion_5seed_results.csv")

assert resnet_path.exists(), "Missing outputs/resnet_mae_5seed_results.csv"
assert phys_path.exists(), "Missing outputs/physfusion_5seed_results.csv"

resnet = pd.read_csv(resnet_path)
phys = pd.read_csv(phys_path)

all_df = pd.concat([resnet, phys], ignore_index=True)

numeric_cols = [
    "SBP_MAE",
    "DBP_MAE",
    "AVG_MAE",
    "SBP_RMSE",
    "DBP_RMSE",
    "AVG_RMSE",
    "SBP_Pearson",
    "DBP_Pearson",
    "subject_SBP_MAE",
    "subject_DBP_MAE",
    "subject_AVG_MAE",
    "subject_SBP_Pearson",
    "subject_DBP_Pearson",
]

summary = all_df.groupby("model")[numeric_cols].agg(["mean", "std", "min", "max"])

print("\n===== Combined ResNet vs PhysFusion summary =====")
print(summary.to_string())

resnet_avg = resnet["AVG_MAE"].mean()
phys_avg = phys["AVG_MAE"].mean()
resnet_std = resnet["AVG_MAE"].std()
phys_std = phys["AVG_MAE"].std()

diff = resnet_avg - phys_avg

print("\n===== Main comparison =====")
print(f"ResNet AVG_MAE mean:     {resnet_avg:.4f}")
print(f"PhysFusion AVG_MAE mean: {phys_avg:.4f}")
print(f"Difference ResNet - PhysFusion: {diff:.4f}")
print(f"ResNet seed STD: {resnet_std:.4f}")
print(f"PhysFusion seed STD: {phys_std:.4f}")

if diff > 0:
    print("PhysFusion mean is lower than ResNet mean.")
else:
    print("PhysFusion mean is not lower than ResNet mean.")

if abs(diff) < resnet_std:
    print("⚠️ Difference is smaller than ResNet seed noise. Do not claim strong improvement yet.")
else:
    print("Difference is larger than ResNet seed noise, but still confirm with bootstrap CI.")

all_df.to_csv("outputs/resnet_vs_physfusion_5seed_results.csv", index=False)
summary.to_csv("outputs/resnet_vs_physfusion_5seed_summary.csv")

print("\n✅ Saved:")
print("outputs/resnet_vs_physfusion_5seed_results.csv")
print("outputs/resnet_vs_physfusion_5seed_summary.csv")
PY


echo
echo "======================================================"
echo "Final files to send to ChatGPT"
echo "======================================================"

echo
echo "----- outputs/physfusion_5seed_summary.csv -----"
cat outputs/physfusion_5seed_summary.csv

echo
echo "----- outputs/resnet_vs_physfusion_5seed_summary.csv -----"
cat outputs/resnet_vs_physfusion_5seed_summary.csv

echo
echo "✅ ALL DONE"
