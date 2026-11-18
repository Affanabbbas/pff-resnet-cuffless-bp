"""Subject-level paired bootstrap of the five-seed ResNet1D+MAE minus PFF ResNet difference.

For each seed and model, per-subject mean absolute error is computed over that subject's
segments (SBP, DBP, and their average). The paired difference (ResNet minus PFF) is averaged
over the five seeds within each subject, and subjects are then resampled with replacement.
Positive values favour PFF ResNet. Writes a NEW summary file; existing files are not modified.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("outputs")
SEEDS = [2026, 2027, 2028, 2029, 2030]
N_BOOT = 5000
RNG_SEED = 0
OLD_SUMMARY = OUT / "bootstrap_resnet_vs_physfusion_subject_level_summary.json"
NEW_SUMMARY = OUT / "bootstrap_five_seed_summary.json"
COLS = ["subject_id", "segment_id", "true_sbp", "true_dbp", "pred_sbp", "pred_dbp"]


def subject_mae(path):
    df = pd.read_csv(path)
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise KeyError(f"{path}: missing {missing}; columns are {list(df.columns)}")
    g = pd.DataFrame({
        "subject_id": df["subject_id"],
        "SBP": (df["pred_sbp"] - df["true_sbp"]).abs(),
        "DBP": (df["pred_dbp"] - df["true_dbp"]).abs(),
    }).groupby("subject_id").mean()
    g["AVG"] = (g["SBP"] + g["DBP"]) / 2
    return g


diffs = []
for s in SEEDS:
    r = subject_mae(OUT / f"resnet_mae_seed{s}_test_predictions.csv")
    p = subject_mae(OUT / f"physfusion_seed{s}_test_predictions.csv")
    assert r.index.equals(p.index), f"subject mismatch at seed {s}"
    diffs.append(r - p)
d = sum(diffs) / len(diffs)  # per-subject, seed-averaged difference
n = len(d)
rng = np.random.default_rng(RNG_SEED)
idx = rng.integers(0, n, size=(N_BOOT, n))

result = {"n_subjects": int(n), "n_bootstrap": N_BOOT, "rng_seed": RNG_SEED, "seeds": SEEDS,
          "definition": "ResNet1D+MAE minus PFF ResNet; positive favours PFF; per-subject mean "
                        "absolute error averaged over seeds, subjects resampled"}
for k in ["SBP", "DBP", "AVG"]:
    v = d[k].to_numpy()
    boot = v[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_two = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
    result[k] = {"diff": float(v.mean()), "ci95": [float(lo), float(hi)], "p_approx": float(p_two)}
    print(f"{k}: diff={v.mean():+.6f}  95% CI [{lo:+.6f}, {hi:+.6f}]  p~{p_two:.4f}")

NEW_SUMMARY.write_text(json.dumps(result, indent=2))
print(f"\nWrote {NEW_SUMMARY}")
if OLD_SUMMARY.exists():
    print("\nExisting summary for comparison:")
    print(OLD_SUMMARY.read_text())