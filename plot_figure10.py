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
# Bland–Altman panel
# ============================================================

def plot_bland_altman(
    ax,
    reference,
    predicted,
    target,
    panel_label,
):
    """
    Create one subject-level Bland–Altman panel.

    Error definition:
        predicted - reference
    """

    reference = np.asarray(reference, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    valid = np.isfinite(reference) & np.isfinite(predicted)
    reference = reference[valid]
    predicted = predicted[valid]

    means = (reference + predicted) / 2.0
    errors = predicted - reference

    bias = np.mean(errors)
    error_sd = np.std(errors, ddof=1)

    lower_loa = bias - 1.96 * error_sd
    upper_loa = bias + 1.96 * error_sd

    # Subject-level observations
    ax.scatter(
        means,
        errors,
        s=30,
        alpha=0.65,
        edgecolors="none",
        rasterized=True,
    )

    # Zero-error reference line
    ax.axhline(
        0,
        linestyle=":",
        linewidth=1.2,
    )

    # Mean bias
    ax.axhline(
        bias,
        linestyle="-",
        linewidth=1.6,
    )

    # 95% limits of agreement
    ax.axhline(
        upper_loa,
        linestyle="--",
        linewidth=1.5,
    )

    ax.axhline(
        lower_loa,
        linestyle="--",
        linewidth=1.5,
    )

    # Axis labels
    ax.set_xlabel(
        f"Mean of predicted and reference {target} (mmHg)"
    )

    ax.set_ylabel(
        f"Prediction error: predicted − reference {target} (mmHg)"
    )

    ax.set_title(target, fontsize=12)

    ax.grid(
        visible=True,
        linestyle=":",
        linewidth=0.6,
        alpha=0.55,
    )

    # Statistics box moved to bottom-left
    statistics_text = (
        f"Bias = {bias:.2f} mmHg\n"
        f"Upper LoA = {upper_loa:.2f} mmHg\n"
        f"Lower LoA = {lower_loa:.2f} mmHg\n"
        f"$n$ = {len(errors)} subjects"
    )

    ax.text(
        0.04,
        0.05,
        statistics_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "alpha": 0.88,
            "linewidth": 0.8,
        },
    )

    # Panel label
    ax.text(
        -0.10,
        1.04,
        panel_label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
    )

    return {
        "target": target,
        "n_subjects": len(errors),
        "bias": bias,
        "error_sd": error_sd,
        "lower_loa": lower_loa,
        "upper_loa": upper_loa,
    }


# ============================================================
# Main
# ============================================================

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {INPUT_FILE.resolve()}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "subject_id",
        "true_sbp",
        "true_dbp",
        "pred_sbp_ensemble4",
        "pred_dbp_ensemble4",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if df["subject_id"].duplicated().any():
        raise ValueError(
            "Duplicate subject IDs found. "
            "Expected one row per subject."
        )

    numeric_columns = [
        "true_sbp",
        "true_dbp",
        "pred_sbp_ensemble4",
        "pred_dbp_ensemble4",
    ]

    df[numeric_columns] = df[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    invalid_rows = df[numeric_columns].isna().any(axis=1)

    if invalid_rows.any():
        print(
            f"Removing {invalid_rows.sum()} rows containing "
            "invalid numerical values."
        )
        df = df.loc[~invalid_rows].copy()

    print(f"Independent test subjects: {len(df)}")

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10.5, 4.8),
        constrained_layout=True,
    )

    sbp_metrics = plot_bland_altman(
        ax=axes[0],
        reference=df["true_sbp"].to_numpy(),
        predicted=df["pred_sbp_ensemble4"].to_numpy(),
        target="SBP",
        panel_label="(a)",
    )

    dbp_metrics = plot_bland_altman(
        ax=axes[1],
        reference=df["true_dbp"].to_numpy(),
        predicted=df["pred_dbp_ensemble4"].to_numpy(),
        target="DBP",
        panel_label="(b)",
    )

    # Output paths
    pdf_path = OUTPUT_DIR / "figure10_ensemble4_bland_altman.pdf"
    png_path = OUTPUT_DIR / "figure10_ensemble4_bland_altman.png"
    tiff_path = OUTPUT_DIR / "figure10_ensemble4_bland_altman.tiff"
    metrics_path = OUTPUT_DIR / "figure10_ensemble4_bland_altman_metrics.csv"

    # Vector format
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    # High-resolution raster formats
    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        tiff_path,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

    metrics_df = pd.DataFrame(
        [sbp_metrics, dbp_metrics]
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    print("\nBland–Altman statistics:")
    print(metrics_df.to_string(index=False))

    print("\nFiles saved:")
    print(pdf_path)
    print(png_path)
    print(tiff_path)
    print(metrics_path)


if __name__ == "__main__":
    main()