from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_DIR = Path("outputs")
FOLDS = [0, 1, 2, 3, 4]


# ============================================================
# Helpers
# ============================================================

def compute_avg_mae(csv_path: Path) -> dict:
    """Compute SBP MAE, DBP MAE, and AVG MAE from a prediction CSV."""

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"true_sbp", "true_dbp", "pred_sbp", "pred_dbp"}
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"{csv_path.name} is missing columns: {sorted(missing)}"
        )

    for col in ["true_sbp", "true_dbp", "pred_sbp", "pred_dbp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["true_sbp", "true_dbp", "pred_sbp", "pred_dbp"]).copy()

    sbp_mae = np.mean(np.abs(df["pred_sbp"] - df["true_sbp"]))
    dbp_mae = np.mean(np.abs(df["pred_dbp"] - df["true_dbp"]))
    avg_mae = (sbp_mae + dbp_mae) / 2.0

    return {
        "SBP_MAE": float(sbp_mae),
        "DBP_MAE": float(dbp_mae),
        "AVG_MAE": float(avg_mae),
        "n_segments": int(len(df)),
    }


# ============================================================
# Main
# ============================================================

def main():
    records = []

    for fold in FOLDS:
        resnet_file = BASE_DIR / f"group5_resnet_fold{fold}_seed2026_test_predictions.csv"
        physfusion_file = BASE_DIR / f"group5_physfusion_fold{fold}_seed2026_test_predictions.csv"

        resnet_metrics = compute_avg_mae(resnet_file)
        physfusion_metrics = compute_avg_mae(physfusion_file)

        records.append({
            "fold": fold + 1,
            "model": "ResNet1D + MAE",
            **resnet_metrics
        })

        records.append({
            "fold": fold + 1,
            "model": "PhysFusion",
            **physfusion_metrics
        })

    df = pd.DataFrame(records)

    # Save fold-wise metrics
    csv_path = OUTPUT_DIR / "figure14_groupcv_foldwise_metrics.csv"
    df.to_csv(csv_path, index=False)

    # Pivot for plotting
    pivot_df = df.pivot(index="fold", columns="model", values="AVG_MAE")

    # Check if PhysFusion is lower in all folds
    physfusion_lower_all = np.all(
        pivot_df["PhysFusion"].to_numpy() < pivot_df["ResNet1D + MAE"].to_numpy()
    )

    print("\nFold-wise AVG MAE:")
    print(pivot_df)

    print("\nPhysFusion lower in all five folds:", physfusion_lower_all)

    summary_records = []
    for model_name in ["ResNet1D + MAE", "PhysFusion"]:
        vals = df.loc[df["model"] == model_name, "AVG_MAE"].to_numpy()
        summary_records.append({
            "model": model_name,
            "mean_AVG_MAE": float(np.mean(vals)),
            "sd_AVG_MAE": float(np.std(vals, ddof=1)),
            "min_AVG_MAE": float(np.min(vals)),
            "max_AVG_MAE": float(np.max(vals)),
        })

    summary_df = pd.DataFrame(summary_records)
    summary_csv = OUTPUT_DIR / "figure14_groupcv_foldwise_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\nSummary:")
    print(summary_df.to_string(index=False))

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)

    x = np.arange(len(pivot_df.index))
    width = 0.36

    bars1 = ax.bar(
        x - width / 2,
        pivot_df["ResNet1D + MAE"].to_numpy(),
        width=width,
        label="ResNet1D + MAE",
        edgecolor="black",
        linewidth=0.6,
        alpha=0.85,
    )

    bars2 = ax.bar(
        x + width / 2,
        pivot_df["PhysFusion"].to_numpy(),
        width=width,
        label="PhysFusion",
        edgecolor="black",
        linewidth=0.6,
        alpha=0.85,
    )

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.05,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i}" for i in pivot_df.index])

    ax.set_xlabel("Held-out subject fold")
    ax.set_ylabel("AVG MAE (mmHg)")
    ax.set_title("Subject-wise Group-CV Fold-wise Comparison", fontsize=11)

    ax.grid(
        visible=True,
        axis="y",
        linestyle=":",
        linewidth=0.6,
        alpha=0.5,
    )

    ax.legend(frameon=True)

    # Save figure
    pdf_path = OUTPUT_DIR / "figure14_groupcv_foldwise_comparison.pdf"
    png_path = OUTPUT_DIR / "figure14_groupcv_foldwise_comparison.png"
    tiff_path = OUTPUT_DIR / "figure14_groupcv_foldwise_comparison.tiff"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(tiff_path, dpi=600, bbox_inches="tight")

    plt.close(fig)

    print("\nFiles saved:")
    print(pdf_path)
    print(png_path)
    print(tiff_path)
    print(csv_path)
    print(summary_csv)


if __name__ == "__main__":
    main()
