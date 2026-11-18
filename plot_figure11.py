from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = Path("outputs/ensemble4_subject_predictions.csv")
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Histogram panel
# ============================================================

def plot_error_histogram(
    ax,
    errors,
    target,
    panel_label,
    bin_width=2.0,
):
    errors = np.asarray(errors, dtype=float)
    errors = errors[np.isfinite(errors)]

    mae = np.mean(np.abs(errors))
    bias = np.mean(errors)
    sd = np.std(errors, ddof=1)

    min_edge = np.floor(errors.min() / bin_width) * bin_width
    max_edge = np.ceil(errors.max() / bin_width) * bin_width + bin_width
    bins = np.arange(min_edge, max_edge + bin_width, bin_width)

    ax.hist(
        errors,
        bins=bins,
        alpha=0.75,
        edgecolor="black",
        linewidth=0.6,
    )

    # Zero-error line
    ax.axvline(
        0,
        linestyle=":",
        linewidth=1.4,
        label="Zero error",
    )

    # Mean error (bias)
    ax.axvline(
        bias,
        linestyle="--",
        linewidth=1.6,
        label="Mean error",
    )

    ax.set_xlabel(f"{target} prediction error (mmHg)\n(predicted − reference)")
    ax.set_ylabel("Number of subjects")
    ax.set_title(target, fontsize=11)

    ax.grid(
        visible=True,
        linestyle=":",
        linewidth=0.6,
        alpha=0.50,
    )

    text = (
        f"Mean error = {bias:.2f} mmHg\n"
        f"MAE = {mae:.2f} mmHg\n"
        f"SD = {sd:.2f} mmHg\n"
        f"$n$ = {len(errors)} subjects"
    )

    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "alpha": 0.88,
            "linewidth": 0.7,
        },
    )

    ax.text(
        -0.10,
        1.05,
        panel_label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
    )

    return {
        "target": target,
        "n_subjects": len(errors),
        "mean_error": bias,
        "mae": mae,
        "sd": sd,
        "min_error": float(np.min(errors)),
        "max_error": float(np.max(errors)),
    }


# ============================================================
# Main
# ============================================================

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Prediction file not found: {INPUT_FILE.resolve()}")

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "subject_id",
        "true_sbp",
        "true_dbp",
        "pred_sbp_ensemble4",
        "pred_dbp_ensemble4",
    }

    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df["subject_id"].duplicated().any():
        raise ValueError("Duplicate subject IDs found. Expected one row per subject.")

    numeric_columns = [
        "true_sbp",
        "true_dbp",
        "pred_sbp_ensemble4",
        "pred_dbp_ensemble4",
    ]

    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=numeric_columns).copy()

    df["err_sbp"] = df["pred_sbp_ensemble4"] - df["true_sbp"]
    df["err_dbp"] = df["pred_dbp_ensemble4"] - df["true_dbp"]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(8.8, 4.0),
        constrained_layout=True,
    )

    sbp_stats = plot_error_histogram(
        ax=axes[0],
        errors=df["err_sbp"].to_numpy(),
        target="SBP",
        panel_label="(a)",
        bin_width=2.0,
    )

    dbp_stats = plot_error_histogram(
        ax=axes[1],
        errors=df["err_dbp"].to_numpy(),
        target="DBP",
        panel_label="(b)",
        bin_width=2.0,
    )

    pdf_path = OUTPUT_DIR / "figure11_error_histograms.pdf"
    png_path = OUTPUT_DIR / "figure11_error_histograms.png"
    tiff_path = OUTPUT_DIR / "figure11_error_histograms.tiff"
    csv_path = OUTPUT_DIR / "figure11_error_histogram_metrics.csv"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(tiff_path, dpi=600, bbox_inches="tight")

    plt.close(fig)

    metrics_df = pd.DataFrame([sbp_stats, dbp_stats])
    metrics_df.to_csv(csv_path, index=False)

    print("\nFigure 11 statistics:")
    print(metrics_df.to_string(index=False))

    print("\nFiles saved:")
    print(pdf_path)
    print(png_path)
    print(tiff_path)
    print(csv_path)


if __name__ == "__main__":
    main()

