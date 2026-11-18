from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = Path("outputs/ensemble4_subject_predictions.csv")
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 10_000
RANDOM_SEED = 2026


# ============================================================
# Bootstrap confidence interval for Pearson correlation
# ============================================================

def bootstrap_pearson_ci(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
    n_bootstrap: int = 10_000,
    seed: int = 2026,
) -> tuple[float, float]:
    """Calculate a subject-level bootstrap 95% CI for Pearson r."""

    rng = np.random.default_rng(seed)
    n_subjects = len(true_values)
    bootstrap_correlations = []

    for _ in range(n_bootstrap):
        indices = rng.integers(
            low=0,
            high=n_subjects,
            size=n_subjects,
        )

        sampled_true = true_values[indices]
        sampled_predicted = predicted_values[indices]

        # Correlation is undefined for constant bootstrap samples.
        if (
            np.std(sampled_true) == 0
            or np.std(sampled_predicted) == 0
        ):
            continue

        bootstrap_correlations.append(
            np.corrcoef(
                sampled_true,
                sampled_predicted,
            )[0, 1]
        )

    if not bootstrap_correlations:
        return np.nan, np.nan

    lower, upper = np.percentile(
        bootstrap_correlations,
        [2.5, 97.5],
    )

    return float(lower), float(upper)


# ============================================================
# Individual scatter-plot panel
# ============================================================

def plot_panel(
    ax: plt.Axes,
    true_values: np.ndarray,
    predicted_values: np.ndarray,
    target: str,
    panel_label: str,
) -> dict:
    """Create one predicted-versus-reference scatter panel."""

    valid = (
        np.isfinite(true_values)
        & np.isfinite(predicted_values)
    )

    true_values = true_values[valid]
    predicted_values = predicted_values[valid]

    pearson_r, p_value = pearsonr(
        true_values,
        predicted_values,
    )

    ci_lower, ci_upper = bootstrap_pearson_ci(
        true_values=true_values,
        predicted_values=predicted_values,
        n_bootstrap=N_BOOTSTRAP,
        seed=RANDOM_SEED,
    )

    errors = predicted_values - true_values
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    bias = np.mean(errors)

    axis_min = min(
        true_values.min(),
        predicted_values.min(),
    )

    axis_max = max(
        true_values.max(),
        predicted_values.max(),
    )

    padding = 0.05 * (axis_max - axis_min)

    if padding == 0:
        padding = 1.0

    lower_limit = axis_min - padding
    upper_limit = axis_max + padding

    # Subject-level observations
    ax.scatter(
        true_values,
        predicted_values,
        s=28,
        alpha=0.65,
        edgecolors="none",
        rasterized=True,
    )

    # Identity line: perfect agreement
    ax.plot(
        [lower_limit, upper_limit],
        [lower_limit, upper_limit],
        linestyle="--",
        linewidth=1.4,
        label="Identity line",
    )

    ax.set_xlim(lower_limit, upper_limit)
    ax.set_ylim(lower_limit, upper_limit)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(f"Reference {target} (mmHg)")
    ax.set_ylabel(f"Predicted {target} (mmHg)")

    ax.grid(
        visible=True,
        linestyle=":",
        linewidth=0.6,
        alpha=0.55,
    )

    statistics = (
        f"$r$ = {pearson_r:.3f}\n"
        f"95% CI [{ci_lower:.3f}, {ci_upper:.3f}]\n"
        f"MAE = {mae:.2f} mmHg\n"
        f"RMSE = {rmse:.2f} mmHg\n"
        f"$n$ = {len(true_values)} subjects"
    )

    ax.text(
        0.04,
        0.96,
        statistics,
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

    ax.set_title(target, fontsize=11)

    return {
        "target": target,
        "n": len(true_values),
        "pearson_r": pearson_r,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
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
        duplicated = df.loc[
            df["subject_id"].duplicated(),
            "subject_id",
        ].tolist()

        raise ValueError(
            "Duplicate subject IDs were found: "
            + ", ".join(map(str, duplicated[:10]))
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

    missing_rows = df[numeric_columns].isna().any(axis=1)

    if missing_rows.any():
        print(
            f"Removing {missing_rows.sum()} rows containing "
            "missing or invalid numerical values."
        )
        df = df.loc[~missing_rows].copy()

    print(f"Prediction file: {INPUT_FILE}")
    print(f"Independent test subjects: {len(df)}")

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(8.5, 4.2),
        constrained_layout=True,
    )

    sbp_results = plot_panel(
        ax=axes[0],
        true_values=df["true_sbp"].to_numpy(dtype=float),
        predicted_values=df[
            "pred_sbp_ensemble4"
        ].to_numpy(dtype=float),
        target="SBP",
        panel_label="(a)",
    )

    dbp_results = plot_panel(
        ax=axes[1],
        true_values=df["true_dbp"].to_numpy(dtype=float),
        predicted_values=df[
            "pred_dbp_ensemble4"
        ].to_numpy(dtype=float),
        target="DBP",
        panel_label="(b)",
    )

    # Do not add a large figure title inside the image.
    # The manuscript caption provides the figure description.

    pdf_path = OUTPUT_DIR / "figure9_ensemble4_scatter.pdf"
    png_path = OUTPUT_DIR / "figure9_ensemble4_scatter.png"
    tiff_path = OUTPUT_DIR / "figure9_ensemble4_scatter.tiff"
    metrics_path = OUTPUT_DIR / "figure9_ensemble4_metrics.csv"

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

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
        [sbp_results, dbp_results]
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    print("\nFigure 9 statistics:")
    print(metrics_df.to_string(index=False))

    print("\nFiles saved:")
    print(pdf_path)
    print(png_path)
    print(tiff_path)
    print(metrics_path)


if __name__ == "__main__":
    main()