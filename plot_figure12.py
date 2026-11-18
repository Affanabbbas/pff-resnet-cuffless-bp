from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = Path("outputs/resnet_vs_physfusion_5seed_results.csv")
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helper functions
# ============================================================

def find_column(columns, candidates):
    """Return the first matching column from a candidate list."""
    lowered = {c.lower().strip(): c for c in columns}
    for candidate in candidates:
        if candidate.lower().strip() in lowered:
            return lowered[candidate.lower().strip()]
    return None


def normalize_model_name(name: str) -> str:
    """Map raw model names to clean labels."""
    name_l = str(name).lower()

    if "physfusion" in name_l:
        return "PhysFusion"
    if "resnet" in name_l:
        return "ResNet1D + MAE"

    return str(name)


# ============================================================
# Main
# ============================================================

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"File not found: {INPUT_FILE.resolve()}")

    df = pd.read_csv(INPUT_FILE)

    print("Columns found:")
    print(df.columns.tolist())

    seed_col = find_column(
        df.columns,
        ["seed", "random_seed"]
    )

    model_col = find_column(
        df.columns,
        ["model", "method", "architecture"]
    )

    avg_mae_col = find_column(
        df.columns,
        [
            "avg_mae",
            "avg mae",
            "average_mae",
            "mean_avg_mae",
            "avg_mae_mmhg",
            "avg_mae_mmHg"
        ]
    )

    if seed_col is None or model_col is None or avg_mae_col is None:
        raise ValueError(
            "Could not detect required columns automatically.\n"
            f"Detected seed_col={seed_col}, model_col={model_col}, avg_mae_col={avg_mae_col}"
        )

    # Keep only relevant columns
    plot_df = df[[seed_col, model_col, avg_mae_col]].copy()
    plot_df.columns = ["seed", "model", "avg_mae"]

    plot_df["model"] = plot_df["model"].apply(normalize_model_name)

    # Keep only the two target models
    plot_df = plot_df[
        plot_df["model"].isin(["ResNet1D + MAE", "PhysFusion"])
    ].copy()

    # Sort seeds
    plot_df["seed"] = pd.to_numeric(plot_df["seed"], errors="coerce")
    plot_df["avg_mae"] = pd.to_numeric(plot_df["avg_mae"], errors="coerce")
    plot_df = plot_df.dropna(subset=["seed", "avg_mae"])
    plot_df["seed"] = plot_df["seed"].astype(int)

    plot_df = plot_df.sort_values(["seed", "model"]).reset_index(drop=True)

    seeds = sorted(plot_df["seed"].unique().tolist())
    expected_seeds = [2026, 2027, 2028, 2029, 2030]

    print("\nDetected seeds:", seeds)

    # Pivot for plotting
    pivot_df = plot_df.pivot(index="seed", columns="model", values="avg_mae")

    # Reindex to expected seed order if available
    if all(seed in pivot_df.index for seed in expected_seeds):
        pivot_df = pivot_df.reindex(expected_seeds)

    # Compute summary stats
    summary_df = (
        plot_df.groupby("model", as_index=False)
        .agg(
            mean_avg_mae=("avg_mae", "mean"),
            sd_avg_mae=("avg_mae", "std"),
            n_seeds=("avg_mae", "count"),
        )
    )

    print("\nSummary statistics:")
    print(summary_df.to_string(index=False))

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.0, 4.4), constrained_layout=True)

    x = np.arange(len(pivot_df.index))
    width = 0.36

    model_order = ["ResNet1D + MAE", "PhysFusion"]

    for i, model_name in enumerate(model_order):
        if model_name not in pivot_df.columns:
            continue

        values = pivot_df[model_name].to_numpy(dtype=float)

        offset = (-width / 2) if i == 0 else (width / 2)

        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=model_name,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.85,
        )

        # Value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.08,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([str(seed) for seed in pivot_df.index])

    ax.set_xlabel("Random seed")
    ax.set_ylabel("AVG MAE (mmHg)")
    ax.set_title("Per-seed Performance Variability", fontsize=11)

    ax.grid(
        visible=True,
        axis="y",
        linestyle=":",
        linewidth=0.6,
        alpha=0.5,
    )

    ax.legend(frameon=True)

    # Save files
    pdf_path = OUTPUT_DIR / "figure12_per_seed_variability.pdf"
    png_path = OUTPUT_DIR / "figure12_per_seed_variability.png"
    tiff_path = OUTPUT_DIR / "figure12_per_seed_variability.tiff"
    csv_path = OUTPUT_DIR / "figure12_per_seed_variability_summary.csv"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(tiff_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    summary_df.to_csv(csv_path, index=False)

    print("\nFiles saved:")
    print(pdf_path)
    print(png_path)
    print(tiff_path)
    print(csv_path)


if __name__ == "__main__":
    main()
