from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs"
FIGDIR = ROOT / "figures" / "paper_plots"
FIGDIR.mkdir(parents=True, exist_ok=True)

PFF_LABEL = "PFF-ResNet"

def savefig(name):
    png = FIGDIR / f"{name}.png"
    pdf = FIGDIR / f"{name}.pdf"
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")

def normalize_name(x):
    return re.sub(r"[^a-z0-9]", "", str(x).lower())

def find_col(df, candidates):
    norm_map = {normalize_name(c): c for c in df.columns}
    for cand in candidates:
        key = normalize_name(cand)
        if key in norm_map:
            return norm_map[key]
    return None

def safe_read_csv(path):
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return None

# ============================================================
# 1. Fixed test split model comparison
# ============================================================
fixed_rows = [
    {
        "Model": "Mean predictor",
        "SBP_MAE": 14.2185,
        "DBP_MAE": 9.3600,
        "AVG_MAE": 11.7893,
        "Subject_AVG_MAE": np.nan,
    },
    {
        "Model": "ResNet1D + MAE",
        "SBP_MAE": 12.9860,
        "DBP_MAE": 8.6038,
        "AVG_MAE": 10.7949,
        "Subject_AVG_MAE": 7.8154,
    },
    {
        "Model": PFF_LABEL,
        "SBP_MAE": 12.9006,
        "DBP_MAE": 8.6265,
        "AVG_MAE": 10.7636,
        "Subject_AVG_MAE": 7.7835,
    },
    {
        "Model": "ResNet1D + MAE-Pearson",
        "SBP_MAE": 13.0546,
        "DBP_MAE": 8.5813,
        "AVG_MAE": 10.8179,
        "Subject_AVG_MAE": 7.6906,
    },
    {
        "Model": "ResNet1D + MAE-Pearson-STD",
        "SBP_MAE": 13.0071,
        "DBP_MAE": 8.7019,
        "AVG_MAE": 10.8545,
        "Subject_AVG_MAE": 7.8827,
    },
    {
        "Model": "CNN-Mamba-Attention",
        "SBP_MAE": 13.2746,
        "DBP_MAE": 8.6139,
        "AVG_MAE": 10.9442,
        "Subject_AVG_MAE": 7.9917,
    },
    {
        "Model": "4-model ensemble",
        "SBP_MAE": 12.7824,
        "DBP_MAE": 8.3697,
        "AVG_MAE": 10.5761,
        "Subject_AVG_MAE": 7.6811,
    },
]

fixed_df = pd.DataFrame(fixed_rows)

plt.figure(figsize=(9, 5))
plot_df = fixed_df.sort_values("AVG_MAE", ascending=True)
plt.barh(plot_df["Model"], plot_df["AVG_MAE"])
plt.xlabel("AVG MAE (mmHg)")
plt.title("Fixed MIMIC-BP test split performance")
plt.gca().invert_yaxis()
for i, v in enumerate(plot_df["AVG_MAE"]):
    plt.text(v + 0.02, i, f"{v:.4f}", va="center", fontsize=8)
savefig("fig_fixed_test_avg_mae_comparison")

plt.figure(figsize=(10, 5))
x = np.arange(len(fixed_df))
w = 0.25
plt.bar(x - w, fixed_df["SBP_MAE"], width=w, label="SBP MAE")
plt.bar(x, fixed_df["DBP_MAE"], width=w, label="DBP MAE")
plt.bar(x + w, fixed_df["AVG_MAE"], width=w, label="AVG MAE")
plt.xticks(x, fixed_df["Model"], rotation=35, ha="right")
plt.ylabel("MAE (mmHg)")
plt.title("SBP, DBP, and AVG MAE on fixed test split")
plt.legend()
savefig("fig_fixed_test_sbp_dbp_avg_mae")

# ============================================================
# 2. ResNet1D and PFF-ResNet 5-seed plots
# ============================================================
resnet_path = OUTDIR / "resnet_mae_5seed_results.csv"
pff_path = OUTDIR / "physfusion_5seed_results.csv"

resnet_df = safe_read_csv(resnet_path)
pff_df = safe_read_csv(pff_path)

if resnet_df is None:
    resnet_df = pd.DataFrame({
        "seed": [2026, 2027, 2028, 2029, 2030],
        "SBP_MAE": [13.1503, 13.0752, 12.9136, 12.9609, 13.1413],
        "DBP_MAE": [8.8505, 8.4685, 8.5337, 8.6588, 8.8629],
        "AVG_MAE": [11.0004, 10.7719, 10.7237, 10.8099, 11.0021],
        "subject_AVG_MAE": [8.1565, 7.7759, 7.8496, 7.8789, 8.0727],
    })

if pff_df is None:
    pff_df = pd.DataFrame({
        "seed": [2026, 2027, 2028, 2029, 2030],
        "SBP_MAE": [12.9722, 12.9091, 12.9432, 12.8719, 12.8754],
        "DBP_MAE": [8.7424, 8.6996, 8.6740, 8.4225, 8.5868],
        "AVG_MAE": [10.8573, 10.8043, 10.8086, 10.6472, 10.7311],
        "subject_AVG_MAE": [7.8780, 7.9959, 7.8458, 7.6554, 7.6784],
    })

seed_col_r = find_col(resnet_df, ["seed", "Seed"])
avg_col_r = find_col(resnet_df, ["AVG_MAE", "avg_mae"])
subavg_col_r = find_col(resnet_df, ["subject_AVG_MAE", "subject_avg_mae", "Subject_AVG_MAE"])

seed_col_p = find_col(pff_df, ["seed", "Seed"])
avg_col_p = find_col(pff_df, ["AVG_MAE", "avg_mae"])
subavg_col_p = find_col(pff_df, ["subject_AVG_MAE", "subject_avg_mae", "Subject_AVG_MAE"])

plt.figure(figsize=(7, 4.5))
plt.plot(resnet_df[seed_col_r], resnet_df[avg_col_r], marker="o", label="ResNet1D + MAE")
plt.plot(pff_df[seed_col_p], pff_df[avg_col_p], marker="o", label=PFF_LABEL)
plt.xlabel("Seed")
plt.ylabel("AVG MAE (mmHg)")
plt.title("Five-seed AVG MAE variability")
plt.legend()
plt.grid(True, alpha=0.3)
savefig("fig_5seed_avg_mae_variability")

plt.figure(figsize=(6.5, 4.5))
models = ["ResNet1D + MAE", PFF_LABEL]
means = [resnet_df[avg_col_r].mean(), pff_df[avg_col_p].mean()]
stds = [resnet_df[avg_col_r].std(ddof=1), pff_df[avg_col_p].std(ddof=1)]
plt.bar(models, means, yerr=stds, capsize=6)
plt.ylabel("AVG MAE (mmHg)")
plt.title("Five-seed mean ± SD comparison")
for i, (m, s) in enumerate(zip(means, stds)):
    plt.text(i, m + s + 0.03, f"{m:.4f} ± {s:.4f}", ha="center", fontsize=8)
savefig("fig_5seed_mean_sd_avg_mae")

if subavg_col_r is not None and subavg_col_p is not None:
    plt.figure(figsize=(6.5, 4.5))
    sub_means = [resnet_df[subavg_col_r].mean(), pff_df[subavg_col_p].mean()]
    sub_stds = [resnet_df[subavg_col_r].std(ddof=1), pff_df[subavg_col_p].std(ddof=1)]
    plt.bar(models, sub_means, yerr=sub_stds, capsize=6)
    plt.ylabel("Subject AVG MAE (mmHg)")
    plt.title("Five-seed subject-level mean ± SD comparison")
    for i, (m, s) in enumerate(zip(sub_means, sub_stds)):
        plt.text(i, m + s + 0.03, f"{m:.4f} ± {s:.4f}", ha="center", fontsize=8)
    savefig("fig_5seed_subject_avg_mae_mean_sd")

# ============================================================
# 3. Subject-wise 5-fold group-CV plots
# ============================================================
folds = np.array([0, 1, 2, 3, 4])

group_resnet_avg = np.array([11.3758, 11.0137, 11.2008, 10.7597, 11.1960])
group_pff_avg = np.array([11.0078, 10.9075, 10.9566, 10.6374, 11.0847])

plt.figure(figsize=(7, 4.5))
plt.plot(folds, group_resnet_avg, marker="o", label="ResNet1D + MAE")
plt.plot(folds, group_pff_avg, marker="o", label=PFF_LABEL)
plt.xticks(folds)
plt.xlabel("Group-CV fold")
plt.ylabel("AVG MAE (mmHg)")
plt.title("Subject-wise 5-fold group-CV AVG MAE")
plt.legend()
plt.grid(True, alpha=0.3)
savefig("fig_group5_cv_foldwise_avg_mae")

plt.figure(figsize=(6.5, 4.5))
models = ["ResNet1D + MAE", PFF_LABEL]
means = [11.1092, 10.9188]
stds = [0.2336, 0.1704]
plt.bar(models, means, yerr=stds, capsize=6)
plt.ylabel("AVG MAE (mmHg)")
plt.title("Group-CV mean ± SD AVG MAE")
for i, (m, s) in enumerate(zip(means, stds)):
    plt.text(i, m + s + 0.03, f"{m:.4f} ± {s:.4f}", ha="center", fontsize=8)
savefig("fig_group5_cv_mean_sd_avg_mae")

# ============================================================
# 4. Bootstrap confidence interval plot
# ============================================================
bootstrap_rows = pd.DataFrame({
    "Comparison": ["Fixed test split", "Group-CV held-out subjects"],
    "Mean": [0.0919, 0.1905],
    "Low": [-0.0375, 0.1032],
    "High": [0.2192, 0.2784],
})

plt.figure(figsize=(7, 3.8))
y = np.arange(len(bootstrap_rows))
means = bootstrap_rows["Mean"].values
lower = means - bootstrap_rows["Low"].values
upper = bootstrap_rows["High"].values - means

plt.errorbar(means, y, xerr=[lower, upper], fmt="o", capsize=6)
plt.axvline(0, linestyle="--", linewidth=1)
plt.yticks(y, bootstrap_rows["Comparison"])
plt.xlabel(f"AVG MAE improvement: ResNet1D - {PFF_LABEL} (mmHg)")
plt.title("Subject-level bootstrap confidence intervals")

for i, row in bootstrap_rows.iterrows():
    plt.text(
        row["High"] + 0.01,
        i,
        f"{row['Mean']:.4f} [{row['Low']:.4f}, {row['High']:.4f}]",
        va="center",
        fontsize=8,
    )

savefig("fig_bootstrap_ci_resnet_minus_pff_resnet")

# ============================================================
# 5. Prediction-based plots
# ============================================================
def pick_prediction_file():
    candidates = []
    patterns = [
        "*ensemble*test_predictions.csv",
        "*physfusion*test_predictions.csv",
        "*resnet*test_predictions.csv",
        "*test_predictions.csv",
    ]
    for pat in patterns:
        candidates.extend(sorted(OUTDIR.glob(pat)))

    seen = set()
    unique = []
    for p in candidates:
        if p not in seen:
            unique.append(p)
            seen.add(p)

    return unique[0] if unique else None

def get_prediction_columns(df):
    true_sbp = find_col(df, ["true_sbp", "sbp_true", "SBP_true", "target_sbp", "label_sbp", "y_sbp", "sbp"])
    pred_sbp = find_col(df, ["pred_sbp", "sbp_pred", "SBP_pred", "predicted_sbp", "prediction_sbp", "yhat_sbp"])
    true_dbp = find_col(df, ["true_dbp", "dbp_true", "DBP_true", "target_dbp", "label_dbp", "y_dbp", "dbp"])
    pred_dbp = find_col(df, ["pred_dbp", "dbp_pred", "DBP_pred", "predicted_dbp", "prediction_dbp", "yhat_dbp"])
    return true_sbp, pred_sbp, true_dbp, pred_dbp

pred_file = pick_prediction_file()

if pred_file is None:
    print("WARNING: No prediction CSV found. Scatter/Bland-Altman/error histogram plots were skipped.")
else:
    print(f"Using prediction file for scatter/Bland-Altman/error plots: {pred_file}")
    pred_df = pd.read_csv(pred_file)
    true_sbp, pred_sbp, true_dbp, pred_dbp = get_prediction_columns(pred_df)

    print("Detected prediction columns:")
    print("true_sbp:", true_sbp, "pred_sbp:", pred_sbp)
    print("true_dbp:", true_dbp, "pred_dbp:", pred_dbp)

    if all(c is not None for c in [true_sbp, pred_sbp, true_dbp, pred_dbp]):
        scatter_df = pred_df.copy()

        if len(scatter_df) > 4000:
            scatter_df = scatter_df.sample(4000, random_state=2026)

        def scatter_plot(true_col, pred_col, target):
            t = scatter_df[true_col].astype(float).values
            p = scatter_df[pred_col].astype(float).values
            lo = min(np.nanmin(t), np.nanmin(p))
            hi = max(np.nanmax(t), np.nanmax(p))

            plt.figure(figsize=(5, 5))
            plt.scatter(t, p, s=8, alpha=0.35)
            plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
            plt.xlabel(f"Reference {target} (mmHg)")
            plt.ylabel(f"Predicted {target} (mmHg)")
            plt.title(f"Predicted vs reference {target}")
            savefig(f"fig_scatter_{target.lower()}")

        def bland_altman_plot(true_col, pred_col, target):
            t = pred_df[true_col].astype(float).values
            p = pred_df[pred_col].astype(float).values

            mean_bp = (t + p) / 2.0
            diff = p - t
            md = np.nanmean(diff)
            sd = np.nanstd(diff, ddof=1)

            plt.figure(figsize=(6, 4.5))
            plt.scatter(mean_bp, diff, s=8, alpha=0.30)
            plt.axhline(md, linestyle="-", linewidth=1, label=f"Mean diff = {md:.2f}")
            plt.axhline(md + 1.96 * sd, linestyle="--", linewidth=1, label=f"+1.96 SD = {md + 1.96 * sd:.2f}")
            plt.axhline(md - 1.96 * sd, linestyle="--", linewidth=1, label=f"-1.96 SD = {md - 1.96 * sd:.2f}")
            plt.xlabel(f"Mean of reference and predicted {target} (mmHg)")
            plt.ylabel(f"Prediction error for {target} (mmHg)")
            plt.title(f"Bland-Altman plot for {target}")
            plt.legend(fontsize=8)
            savefig(f"fig_bland_altman_{target.lower()}")

        def error_hist_plot(true_col, pred_col, target):
            t = pred_df[true_col].astype(float).values
            p = pred_df[pred_col].astype(float).values
            err = p - t

            plt.figure(figsize=(6, 4.5))
            plt.hist(err, bins=60)
            plt.axvline(np.nanmean(err), linestyle="--", linewidth=1, label=f"Mean error = {np.nanmean(err):.2f}")
            plt.xlabel(f"Prediction error for {target} (mmHg)")
            plt.ylabel("Count")
            plt.title(f"Error distribution for {target}")
            plt.legend(fontsize=8)
            savefig(f"fig_error_hist_{target.lower()}")

        scatter_plot(true_sbp, pred_sbp, "SBP")
        scatter_plot(true_dbp, pred_dbp, "DBP")
        bland_altman_plot(true_sbp, pred_sbp, "SBP")
        bland_altman_plot(true_dbp, pred_dbp, "DBP")
        error_hist_plot(true_sbp, pred_sbp, "SBP")
        error_hist_plot(true_dbp, pred_dbp, "DBP")

    else:
        print("WARNING: Could not detect prediction columns automatically.")
        print("Available columns:")
        print(list(pred_df.columns))
        print("Edit get_prediction_columns() inside scripts/make_paper_plots.py according to these column names.")

print()
print("All available plots saved in:")
print(FIGDIR)
