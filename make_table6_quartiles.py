from pathlib import Path

import pandas as pd

INPUT = Path("outputs/physfusion_seed2026_test_subject_predictions.csv")
LABELS = ["Q1", "Q2", "Q3", "Q4"]
TABLE6 = {
    "SBP": [
        (58, "79.77-104.15", 9.205498206478426),
        (57, "104.47-111.37", 4.953413855123242),
        (57, "111.67-120.56", 5.625285584745351),
        (57, "120.80-151.85", 15.566029946845871),
    ],
    "DBP": [
        (58, "36.97-52.74", 6.898993207120347),
        (57, "52.97-59.37", 3.650365012988709),
        (57, "59.43-65.60", 5.734691211633515),
        (57, "65.93-87.33", 11.383904656750419),
    ],
}

df = pd.read_csv(INPUT)
if df["subject_id"].duplicated().any():
    raise ValueError("Expected one prediction row per subject.")

for target in ("SBP", "DBP"):
    ref, pred = f"true_{target.lower()}", f"pred_{target.lower()}"
    quartile, edges = pd.qcut(df[ref], 4, labels=LABELS, retbins=True)
    work = df.assign(quartile=quartile, error=df[pred] - df[ref])

    print(f"\n{target}: qcut reference edges = {edges}")
    for i, label in enumerate(LABELS):
        group = work.loc[work["quartile"] == label]
        observed_range = f"{group[ref].min():.2f}-{group[ref].max():.2f}"
        mae = group["error"].abs().mean()
        mean_error = group["error"].mean()
        old_n, old_range, old_mae = TABLE6[target][i]
        print(
            f"{label}: edges [{edges[i]:.4f}, {edges[i + 1]:.4f}], "
            f"observed={observed_range}, n={len(group)}, "
            f"MAE={mae:.6f}, mean_error={mean_error:.6f}; "
            f"Table 6 n/range/MAE={old_n}/{old_range}/{old_mae:.6f}; "
            f"MAE difference={mae - old_mae:+.6f}"
        )