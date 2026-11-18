from pathlib import Path

import numpy as np
import pandas as pd


N_SUBJECTS = 229
N_BOOTSTRAP = 10_000
RANDOM_SEED = 2026

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "outputs"

RESNET_FILE = DATA_DIR / "resnet_mae_loss_test_subject_predictions.csv"
PFF_FILE = DATA_DIR / "resnet1d_physfusion_mae_test_subject_predictions.csv"


def main():
    resnet = pd.read_csv(RESNET_FILE)
    pff = pd.read_csv(PFF_FILE)

    required_columns = {
        "subject_id",
        "true_sbp",
        "true_dbp",
        "pred_sbp",
        "pred_dbp",
    }

    for name, frame in [("ResNet1D", resnet), ("PFF ResNet", pff)]:
        missing = required_columns - set(frame.columns)
        if missing:
            raise ValueError(f"{name} file is missing columns: {sorted(missing)}")

        if frame["subject_id"].duplicated().any():
            raise ValueError(f"{name} file contains duplicate subject_id values")

    merged = resnet.merge(
        pff,
        on="subject_id",
        how="inner",
        suffixes=("_resnet", "_pff"),
        validate="one_to_one",
    )

    if len(merged) != N_SUBJECTS:
        raise ValueError(
            f"Expected exactly {N_SUBJECTS} matched subjects, "
            f"found {len(merged)}"
        )

    print(f"Matched subjects: {len(merged)}")

    resnet_sbp_error = (
        merged["pred_sbp_resnet"] - merged["true_sbp_resnet"]
    ).abs()
    pff_sbp_error = (
        merged["pred_sbp_pff"] - merged["true_sbp_pff"]
    ).abs()

    resnet_dbp_error = (
        merged["pred_dbp_resnet"] - merged["true_dbp_resnet"]
    ).abs()
    pff_dbp_error = (
        merged["pred_dbp_pff"] - merged["true_dbp_pff"]
    ).abs()

    resnet_avg_error = (resnet_sbp_error + resnet_dbp_error) / 2
    pff_avg_error = (pff_sbp_error + pff_dbp_error) / 2

    differences = pd.DataFrame(
        {
            "SBP": resnet_sbp_error.to_numpy() - pff_sbp_error.to_numpy(),
            "DBP": resnet_dbp_error.to_numpy() - pff_dbp_error.to_numpy(),
            "AVG": resnet_avg_error.to_numpy() - pff_avg_error.to_numpy(),
        }
    )

    rng = np.random.default_rng(RANDOM_SEED)
    bootstrap_indices = rng.integers(
        0,
        N_SUBJECTS,
        size=(N_BOOTSTRAP, N_SUBJECTS),
    )

    bootstrap_results = {}
    for metric in ["SBP", "DBP", "AVG"]:
        observed = differences[metric].mean()
        bootstrap_means = differences[metric].to_numpy()[
            bootstrap_indices
        ].mean(axis=1)

        ci_lower, ci_upper = np.percentile(
            bootstrap_means,
            [2.5, 97.5],
        )

        p_lower = np.mean(bootstrap_means <= 0)
        p_upper = np.mean(bootstrap_means >= 0)
        p_value = 2 * min(p_lower, p_upper)

        bootstrap_results[metric] = {
            "observed": observed,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": p_value,
        }

    print(f"Bootstrap samples: {N_BOOTSTRAP}")
    print(f"Random seed: {RANDOM_SEED}")
    print()

    for metric, result in bootstrap_results.items():
        print(
            f"{metric}: "
            f"observed difference = {result['observed']:.6f} mmHg, "
            f"95% CI = "
            f"[{result['ci_lower']:.6f}, {result['ci_upper']:.6f}], "
            f"p-value = {result['p_value']:.6f}"
        )

    avg_observed = bootstrap_results["AVG"]["observed"]
    print()
    print(
        "AVG observed difference approximately 0.031940:",
        bool(np.isclose(avg_observed, 0.031940, atol=1e-6)),
    )


if __name__ == "__main__":
    main()
