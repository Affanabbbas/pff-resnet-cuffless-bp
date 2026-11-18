from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack, vstack


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
MODEL_FILES = {
    "resnet_mae_loss": "resnet_mae_loss",
    "cnn_mamba_small": "cnn_mamba_small",
    "resnet_mae_pearson_a005": "resnet_mae_pearson_a005",
    "resnet_mae_pearson_std_p003_s005": "resnet_mae_pearson_std_p003_s005",
}
FIXED_WEIGHTS = np.array([0.15, 0.30, 0.11, 0.44], dtype=float)
KEY_COLUMNS = ["subject_id", "segment_id"]
TRUE_COLUMNS = ["true_sbp", "true_dbp"]
PRED_COLUMNS = ["pred_sbp", "pred_dbp"]


def load_split(split: str) -> pd.DataFrame:
    """Load and validate aligned predictions for all four models."""
    merged: pd.DataFrame | None = None

    for model_name, file_stem in MODEL_FILES.items():
        path = OUTPUTS / f"{file_stem}_{split}_predictions.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {split} predictions: {path}")

        frame = pd.read_csv(path)
        required = set(KEY_COLUMNS + TRUE_COLUMNS + PRED_COLUMNS)
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        if frame.duplicated(KEY_COLUMNS).any():
            raise ValueError(f"{path.name} has duplicate subject/segment keys")

        frame = frame[KEY_COLUMNS + TRUE_COLUMNS + PRED_COLUMNS].copy()
        frame = frame.rename(
            columns={
                "pred_sbp": f"{model_name}_pred_sbp",
                "pred_dbp": f"{model_name}_pred_dbp",
            }
        )

        if merged is None:
            merged = frame
            continue

        merged = merged.merge(
            frame,
            on=KEY_COLUMNS,
            how="outer",
            validate="one_to_one",
            indicator=True,
            suffixes=("", "_other"),
        )
        unmatched = merged["_merge"] != "both"
        if unmatched.any():
            raise ValueError(
                f"{split} model predictions do not share identical segment keys "
                f"({int(unmatched.sum())} unmatched rows)"
            )
        merged = merged.drop(columns="_merge")

        for column in TRUE_COLUMNS:
            other_column = f"{column}_other"
            if not np.allclose(
                merged[column].to_numpy(dtype=float),
                merged[other_column].to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-5,
            ):
                raise ValueError(f"Reference values differ across {split} CSVs: {column}")
            merged = merged.drop(columns=other_column)

    assert merged is not None
    numeric_columns = TRUE_COLUMNS + [
        f"{model_name}_{target}"
        for model_name in MODEL_FILES
        for target in PRED_COLUMNS
    ]
    values = merged[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite reference or prediction values in {split} CSVs")
    return merged


def weighted_predictions(frame: pd.DataFrame, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sbp = np.column_stack(
        [frame[f"{name}_pred_sbp"].to_numpy(dtype=float) for name in MODEL_FILES]
    ) @ weights
    dbp = np.column_stack(
        [frame[f"{name}_pred_dbp"].to_numpy(dtype=float) for name in MODEL_FILES]
    ) @ weights
    return sbp, dbp


def print_metrics(title: str, true_sbp: np.ndarray, true_dbp: np.ndarray,
                  pred_sbp: np.ndarray, pred_dbp: np.ndarray) -> None:
    sbp_mae = np.mean(np.abs(pred_sbp - true_sbp))
    dbp_mae = np.mean(np.abs(pred_dbp - true_dbp))
    avg_mae = (sbp_mae + dbp_mae) / 2.0
    print(f"{title}")
    print(f"  SBP MAE: {sbp_mae:.6f} mmHg")
    print(f"  DBP MAE: {dbp_mae:.6f} mmHg")
    print(f"  AVG MAE: {avg_mae:.6f} mmHg")


def fit_simplex_weights(validation: pd.DataFrame) -> np.ndarray:
    """Minimize mean SBP/DBP absolute error subject to simplex weights."""
    n_rows = len(validation)
    if n_rows == 0:
        raise ValueError("Validation predictions are empty")

    prediction_matrices = []
    targets = []
    for target, suffix in zip(TRUE_COLUMNS, ("pred_sbp", "pred_dbp")):
        prediction_matrices.append(
            np.column_stack(
                [validation[f"{name}_{suffix}"].to_numpy(dtype=float) for name in MODEL_FILES]
            )
        )
        targets.append(validation[target].to_numpy(dtype=float))

    prediction_matrices = [csr_matrix(matrix) for matrix in prediction_matrices]
    slack_identity = eye(n_rows, format="csr")
    zero_block = csr_matrix((n_rows, n_rows))
    zero_weights = csr_matrix((n_rows, len(MODEL_FILES)))

    # Variables are [four weights, SBP absolute errors, DBP absolute errors].
    constraints = []
    bounds = []
    for channel, (predictions, target) in enumerate(zip(prediction_matrices, targets)):
        sbp_slack = -slack_identity if channel == 0 else zero_block
        dbp_slack = -slack_identity if channel == 1 else zero_block
        positive = hstack([predictions, sbp_slack, dbp_slack], format="csr")
        negative = hstack([-predictions, sbp_slack, dbp_slack], format="csr")
        constraints.extend([positive, negative])
        bounds.extend([target, -target])

    a_ub = vstack(constraints, format="csr")
    b_ub = np.concatenate(bounds)
    c = np.concatenate(
        [np.zeros(len(MODEL_FILES)), np.full(2 * n_rows, 1.0 / (2 * n_rows))]
    )
    a_eq = csr_matrix(
        ([1.0] * len(MODEL_FILES), ([0] * len(MODEL_FILES), list(range(len(MODEL_FILES))))),
        shape=(1, len(MODEL_FILES) + 2 * n_rows),
    )
    result = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=np.array([1.0]),
        bounds=[(0.0, None)] * (len(MODEL_FILES) + 2 * n_rows),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Validation weight optimization failed: {result.message}")

    weights = result.x[: len(MODEL_FILES)]
    weights[np.abs(weights) < 1e-10] = 0.0
    weights /= weights.sum()
    return weights


def report_saved_ensemble() -> None:
    subject_path = OUTPUTS / "ensemble4_subject_predictions.csv"
    subject = pd.read_csv(subject_path)
    subject_required = {"true_sbp", "true_dbp", "pred_sbp_ensemble4", "pred_dbp_ensemble4"}
    missing = subject_required.difference(subject.columns)
    if missing:
        raise ValueError(f"{subject_path.name} is missing columns: {sorted(missing)}")
    print_metrics(
        "Part 2: saved ensemble4_subject_predictions.csv (subject-level)",
        subject["true_sbp"].to_numpy(dtype=float),
        subject["true_dbp"].to_numpy(dtype=float),
        subject["pred_sbp_ensemble4"].to_numpy(dtype=float),
        subject["pred_dbp_ensemble4"].to_numpy(dtype=float),
    )

    segment_path = OUTPUTS / "ensemble4_test_predictions.csv"
    if segment_path.is_file():
        segment = pd.read_csv(segment_path)
        segment_required = {
            "true_sbp", "true_dbp", "pred_sbp_ensemble4", "pred_dbp_ensemble4"
        }
        missing = segment_required.difference(segment.columns)
        if missing:
            raise ValueError(f"{segment_path.name} is missing columns: {sorted(missing)}")
        print_metrics(
            "Part 2: saved ensemble4_test_predictions.csv (segment-level)",
            segment["true_sbp"].to_numpy(dtype=float),
            segment["true_dbp"].to_numpy(dtype=float),
            segment["pred_sbp_ensemble4"].to_numpy(dtype=float),
            segment["pred_dbp_ensemble4"].to_numpy(dtype=float),
        )
    else:
        print("Part 2: no saved ensemble4 segment prediction file found")


def main() -> None:
    test = load_split("test")
    true_sbp = test["true_sbp"].to_numpy(dtype=float)
    true_dbp = test["true_dbp"].to_numpy(dtype=float)

    print(f"Aligned test segments: {len(test)}")
    print("Model order:", ", ".join(MODEL_FILES))

    fixed_sbp, fixed_dbp = weighted_predictions(test, FIXED_WEIGHTS)
    print_metrics(
        "Part 1: fixed weights [0.15, 0.30, 0.11, 0.44]",
        true_sbp,
        true_dbp,
        fixed_sbp,
        fixed_dbp,
    )

    report_saved_ensemble()

    validation = load_split("val")
    fitted_weights = fit_simplex_weights(validation)
    fitted_val_sbp, fitted_val_dbp = weighted_predictions(validation, fitted_weights)
    print("Part 3: validation-fitted non-negative simplex weights")
    for name, weight in zip(MODEL_FILES, fitted_weights):
        print(f"  {name}: {weight:.8f}")
    print_metrics(
        "  validation fit objective",
        validation["true_sbp"].to_numpy(dtype=float),
        validation["true_dbp"].to_numpy(dtype=float),
        fitted_val_sbp,
        fitted_val_dbp,
    )
    fitted_test_sbp, fitted_test_dbp = weighted_predictions(test, fitted_weights)
    print_metrics(
        "  test performance with validation-fitted weights",
        true_sbp,
        true_dbp,
        fitted_test_sbp,
        fitted_test_dbp,
    )

    equal_weights = np.full(len(MODEL_FILES), 1.0 / len(MODEL_FILES))
    equal_sbp, equal_dbp = weighted_predictions(test, equal_weights)
    print("Part 3: equal weights")
    for name, weight in zip(MODEL_FILES, equal_weights):
        print(f"  {name}: {weight:.8f}")
    print_metrics(
        "  test performance with equal weights",
        true_sbp,
        true_dbp,
        equal_sbp,
        equal_dbp,
    )


if __name__ == "__main__":
    main()