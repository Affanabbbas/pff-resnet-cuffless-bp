from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
METRIC_NAMES = ("SBP_MAE", "DBP_MAE", "AVG_MAE")
SEEDS = range(2026, 2031)


def read_json(filename: str) -> dict[str, Any]:
    path = OUTPUTS / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required metrics file not found: {path}")
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def extract_row(
    label: str,
    data: dict[str, Any],
    metric_section: str | None = None,
    subject_section: str = "subject_level",
) -> dict[str, Any]:
    metrics = data[metric_section] if metric_section else data
    subject_metrics = data.get(subject_section, {})
    return {
        "label": label,
        **{name: metrics.get(name) for name in METRIC_NAMES},
        "subject_avg": subject_metrics.get("AVG_MAE"),
    }


def read_row(
    label: str,
    filename: str,
    metric_section: str | None = None,
    subject_section: str = "subject_level",
) -> dict[str, Any]:
    return extract_row(
        label,
        read_json(filename),
        metric_section=metric_section,
        subject_section=subject_section,
    )


def summarize_rows(label: str, rows: list[dict[str, Any]], use_sd: bool) -> dict[str, Any]:
    aggregate = stdev if use_sd else mean
    summary: dict[str, Any] = {"label": label}
    for key in (*METRIC_NAMES, "subject_avg"):
        values = [row[key] for row in rows if row[key] is not None]
        summary[key] = aggregate(values) if values else None
    return summary


def format_value(value: Any) -> str:
    return f"{value:.4f}" if value is not None else "N/A"


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = ("Model / run", "SBP MAE", "DBP MAE", "AVG MAE", "Subj AVG")
    body = [
        (
            row["label"],
            *(format_value(row[name]) for name in METRIC_NAMES),
            format_value(row["subject_avg"]),
        )
        for row in rows
    ]
    widths = [
        max(len(headers[column]), *(len(row[column]) for row in body))
        for column in range(len(headers))
    ]
    print(" | ".join(value.ljust(widths[i]) for i, value in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in body:
        print(" | ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def main() -> None:
    rows: list[dict[str, Any]] = []

    rows.append(
        read_row(
            "Mean predictor",
            "mean_predictor_baseline_metrics.json",
            metric_section="test",
        )
    )
    rows.extend(
        [
            read_row("ResNet MAE (seed 42)", "resnet_mae_loss_test_metrics.json"),
            read_row("PFF (seed 42)", "resnet1d_physfusion_mae_test_metrics.json"),
            read_row(
                "ResNet MAE Pearson a=0.05 (seed 42)",
                "resnet_mae_pearson_a005_test_metrics.json",
            ),
            read_row(
                "ResNet MAE Pearson+STD (seed 42)",
                "resnet_mae_pearson_std_p003_s005_test_metrics.json",
            ),
            read_row("CNN-Mamba (seed 42)", "cnn_mamba_small_test_metrics.json"),
            read_row(
                "Ensemble4",
                "ensemble4_test_metrics.json",
                metric_section="segment_level",
            ),
        ]
    )

    for model_label, stem in (
        ("ResNet MAE", "resnet_mae"),
        ("PhysFusion", "physfusion"),
    ):
        seed_rows = [
            read_row(f"{model_label} (seed {seed})", f"{stem}_seed{seed}_test_metrics.json")
            for seed in SEEDS
        ]
        rows.extend(seed_rows)
        rows.append(summarize_rows(f"{model_label} 5-seed mean", seed_rows, use_sd=False))
        rows.append(summarize_rows(f"{model_label} 5-seed SD", seed_rows, use_sd=True))

    for model_label, stem in (("Group5 PFF", "physfusion"), ("Group5 ResNet", "resnet")):
        fold_rows = [
            read_row(
                f"{model_label} fold {fold} (seed 2026)",
                f"group5_{stem}_fold{fold}_seed2026_test_metrics.json",
            )
            for fold in range(5)
        ]
        rows.extend(fold_rows)
        rows.append(summarize_rows(f"{model_label} 5-fold mean", fold_rows, use_sd=False))
        rows.append(summarize_rows(f"{model_label} 5-fold SD", fold_rows, use_sd=True))

    print_table(rows)


if __name__ == "__main__":
    main()
