from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch
import yaml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from datasets.mimic_bp_dataset import MIMICBPDataset, compute_train_label_stats


SEEDS = (2026, 2027, 2028, 2029, 2030)
TARGET_NAMES = ("SBP", "DBP")
EXPECTED_DESCRIPTOR_COUNT = 12
EXPECTED_MEAN_PREDICTOR_AVG_MAE = 11.7893


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train descriptor-only baselines on the PhysFusion dataset splits."
    )
    parser.add_argument("--config-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).parent / "outputs" / "descriptors_only_cache",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "outputs" / "descriptors_only_results.json",
    )
    return parser.parse_args()


def load_seed_configs(config_dir: Path) -> Dict[int, Dict[str, Any]]:
    configs: Dict[int, Dict[str, Any]] = {}
    for seed in SEEDS:
        path = config_dir / f"config_physfusion_seed{seed}.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"Missing config for seed {seed}: {path}")
        with path.open(encoding="utf-8") as file:
            config = yaml.safe_load(file)
        if not config["signals"].get("include_physio_features", False):
            raise ValueError(f"{path} must enable signals.include_physio_features")
        configs[seed] = config

    reference = dataset_config_signature(configs[SEEDS[0]])
    for seed in SEEDS[1:]:
        if dataset_config_signature(configs[seed]) != reference:
            raise ValueError(
                "All five configs must use identical data, split, and signal processing settings."
            )
    return configs


def dataset_config_signature(config: Dict[str, Any]) -> Dict[str, Any]:
    signals = config["signals"]
    signal_keys = (
        "use_derivatives",
        "segment_seconds",
        "sampling_rate",
        "expected_segments_per_subject",
        "normalization",
        "preprocess",
        "filter_signals",
        "quality_filter",
        "min_ppg_peak_count",
        "include_physio_features",
        "max_cache_subjects",
    )
    return {
        "data_dir": str(Path(config["data_dir"]).resolve()),
        "split_dir": str(Path(config["split_dir"]).resolve())
        if config.get("split_dir")
        else None,
        "signals": {key: signals.get(key) for key in signal_keys},
        "label_standardization": bool(
            config["training"].get("label_standardization", False)
        ),
    }


def make_dataset(config: Dict[str, Any], split: str) -> MIMICBPDataset:
    signals = config["signals"]
    return MIMICBPDataset(
        data_dir=config["data_dir"],
        split=split,
        split_dir=config.get("split_dir"),
        use_derivatives=signals.get("use_derivatives", False),
        normalization=signals.get("normalization", "global_train"),
        sampling_rate=signals.get("sampling_rate", 125),
        segment_seconds=signals.get("segment_seconds", 30),
        expected_segments_per_subject=signals.get("expected_segments_per_subject", 30),
        max_cache_subjects=signals.get("max_cache_subjects", 16),
        verbose=True,
        preprocess=signals.get("preprocess", False),
        filter_signals=signals.get("filter_signals", True),
        quality_filter=signals.get("quality_filter", False),
        min_ppg_peak_count=signals.get("min_ppg_peak_count", 10),
        include_physio_features=True,
        standardize_labels=False,
    )


def cache_signature(
    config: Dict[str, Any],
    split: str,
    label_stats: Dict[str, np.ndarray],
) -> str:
    payload = {
        "dataset": dataset_config_signature(config),
        "split": split,
        "label_mean": np.asarray(label_stats["mean"]).tolist(),
        "label_std": np.asarray(label_stats["std"]).tolist(),
        "descriptor_count": EXPECTED_DESCRIPTOR_COUNT,
    }
    return json.dumps(payload, sort_keys=True)


def load_or_create_split_cache(
    config: Dict[str, Any],
    split: str,
    label_stats: Dict[str, np.ndarray],
    cache_dir: Path,
) -> Dict[str, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{split}.npz"
    signature = cache_signature(config, split, label_stats)

    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            if str(cached["signature"].item()) == signature:
                arrays = {
                    key: cached[key].copy()
                    for key in ("features", "labels", "subject_ids", "segment_ids")
                }
                print(f"Loaded {split} descriptor cache: {cache_path}")
                return arrays
        print(f"Cache settings changed; rebuilding {cache_path}")

    dataset = make_dataset(config, split)
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"].get("batch_size", 64)),
        shuffle=False,
        num_workers=0,
    )
    features_parts = []
    labels_parts = []
    subject_ids = []
    segment_ids = []

    for _, labels, meta in loader:
        features_parts.append(meta["physio_features"].cpu().numpy().astype(np.float32))
        labels_parts.append(labels.numpy().astype(np.float32))
        subject_ids.extend(str(subject_id) for subject_id in meta["subject_id"])
        segment_ids.extend(meta["segment_id"].numpy().astype(np.int32).tolist())

    arrays = {
        "features": np.concatenate(features_parts, axis=0),
        "labels": np.concatenate(labels_parts, axis=0),
        "subject_ids": np.asarray(subject_ids, dtype=np.str_),
        "segment_ids": np.asarray(segment_ids, dtype=np.int32),
    }
    if arrays["features"].shape[1] != EXPECTED_DESCRIPTOR_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_DESCRIPTOR_COUNT} descriptors, got "
            f"{arrays['features'].shape[1]}"
        )

    np.savez_compressed(cache_path, signature=signature, **arrays)
    print(f"Cached {split} descriptors: {cache_path} ({len(arrays['labels'])} segments)")
    return arrays


def standardize_labels(labels: np.ndarray, label_stats: Dict[str, np.ndarray]) -> np.ndarray:
    mean = np.asarray(label_stats["mean"], dtype=np.float64)
    scale = np.asarray(label_stats["std"], dtype=np.float64) + 1e-8
    return ((labels.astype(np.float64) - mean) / scale).astype(np.float32)


def inverse_standardize_labels(
    labels: np.ndarray, label_stats: Dict[str, np.ndarray]
) -> np.ndarray:
    mean = np.asarray(label_stats["mean"], dtype=np.float64)
    scale = np.asarray(label_stats["std"], dtype=np.float64) + 1e-8
    return labels.astype(np.float64) * scale + mean


def mae_metrics(truth: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    per_target = np.mean(np.abs(truth - prediction), axis=0)
    return {
        "SBP_MAE": float(per_target[0]),
        "DBP_MAE": float(per_target[1]),
        "AVG_MAE": float(np.mean(per_target)),
    }


def fit_target_models(
    estimator_class: Any,
    params: Dict[str, Any],
    train_features: np.ndarray,
    train_targets: np.ndarray,
    seed: int,
    scale_features: bool,
) -> Tuple[list[Any], StandardScaler | None]:
    scaler = StandardScaler() if scale_features else None
    if scaler is not None:
        train_features = scaler.fit_transform(train_features)

    estimators = []
    for target_index in range(2):
        model_params = dict(params)
        if "random_state" in estimator_class().get_params():
            model_params["random_state"] = seed + target_index
        estimator = estimator_class(**model_params)
        estimator.fit(train_features, train_targets[:, target_index])
        estimators.append(estimator)
    return estimators, scaler


def predict_target_models(
    estimators: Iterable[Any],
    features: np.ndarray,
    scaler: StandardScaler | None,
) -> np.ndarray:
    if scaler is not None:
        features = scaler.transform(features)
    return np.column_stack([estimator.predict(features) for estimator in estimators])


def select_hyperparameters(
    family: str,
    train_x: np.ndarray,
    train_y_std: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    label_stats: Dict[str, np.ndarray],
    seed: int,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    if family == "gradient_boosting":
        candidates = [
            {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "loss": "absolute_error",
            }
            for n_estimators in (100, 300)
            for learning_rate in (0.03, 0.1)
            for max_depth in (2, 3)
            for min_samples_leaf in (5, 20)
        ]
        estimator_class = GradientBoostingRegressor
        scale_features = False
    elif family == "mlp":
        candidates = [
            {
                "hidden_layer_sizes": hidden_layers,
                "alpha": alpha,
                "activation": activation,
                "solver": "adam",
                "max_iter": 500,
                "early_stopping": False,
            }
            for hidden_layers in ((32,), (64, 32))
            for alpha in (1e-4, 1e-2)
            for activation in ("relu", "tanh")
        ]
        estimator_class = MLPRegressor
        scale_features = True
    else:
        raise ValueError(f"Unknown model family: {family}")

    best_params = None
    best_metrics = None
    best_score = float("inf")
    val_y_std = standardize_labels(val_y, label_stats)

    for params in candidates:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            estimators, scaler = fit_target_models(
                estimator_class,
                params,
                train_x,
                train_y_std,
                seed,
                scale_features,
            )
        val_prediction_std = predict_target_models(estimators, val_x, scaler)
        val_prediction = inverse_standardize_labels(val_prediction_std, label_stats)
        metrics = mae_metrics(val_y, val_prediction)
        if metrics["AVG_MAE"] < best_score:
            best_score = metrics["AVG_MAE"]
            best_params = params
            best_metrics = metrics

    assert best_params is not None and best_metrics is not None
    return best_params, best_metrics


def run_seed(
    seed: int,
    config: Dict[str, Any],
    splits: Dict[str, Dict[str, np.ndarray]],
    label_stats: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    train = splits["train"]
    val = splits["val"]
    test = splits["test"]
    train_y_std = standardize_labels(train["labels"], label_stats)
    test_truth = test["labels"].astype(np.float64)
    train_mean = np.asarray(label_stats["mean"], dtype=np.float64)
    mean_prediction = np.broadcast_to(train_mean, test_truth.shape)
    mean_metrics = mae_metrics(test_truth, mean_prediction)

    seed_result: Dict[str, Any] = {
        "mean_predictor": mean_metrics,
        "models": {},
    }
    model_settings = (
        ("gradient_boosting", GradientBoostingRegressor, False),
        ("mlp", MLPRegressor, True),
    )
    for family, estimator_class, scale_features in model_settings:
        params, val_metrics = select_hyperparameters(
            family,
            train["features"],
            train_y_std,
            val["features"],
            val["labels"],
            label_stats,
            seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            estimators, scaler = fit_target_models(
                estimator_class,
                params,
                train["features"],
                train_y_std,
                seed,
                scale_features,
            )
        prediction_std = predict_target_models(estimators, test["features"], scaler)
        prediction = inverse_standardize_labels(prediction_std, label_stats)
        test_metrics = mae_metrics(test_truth, prediction)
        seed_result["models"][family] = {
            "selected_hyperparameters": params,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
        }
        print(
            f"Seed {seed} {family}: test SBP MAE={test_metrics['SBP_MAE']:.4f}, "
            f"DBP MAE={test_metrics['DBP_MAE']:.4f}, "
            f"AVG MAE={test_metrics['AVG_MAE']:.4f}"
        )

    print(
        f"Seed {seed} mean predictor: test SBP MAE={mean_metrics['SBP_MAE']:.4f}, "
        f"DBP MAE={mean_metrics['DBP_MAE']:.4f}, "
        f"AVG MAE={mean_metrics['AVG_MAE']:.4f}"
    )
    return seed_result


def summarize_results(results: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    groups = {
        "mean_predictor": lambda seed_result: seed_result["mean_predictor"],
        "gradient_boosting": lambda seed_result: seed_result["models"]["gradient_boosting"]["test_metrics"],
        "mlp": lambda seed_result: seed_result["models"]["mlp"]["test_metrics"],
    }
    for name, metric_getter in groups.items():
        summary[name] = {}
        for metric in ("SBP_MAE", "DBP_MAE", "AVG_MAE"):
            values = [metric_getter(results[str(seed)])[metric] for seed in SEEDS]
            summary[name][metric] = {
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)),
            }
    return summary


def main() -> None:
    args = parse_args()
    configs = load_seed_configs(args.config_dir)
    first_config = configs[SEEDS[0]]
    train_dataset = make_dataset(first_config, "train")
    if not first_config["training"].get("label_standardization", False):
        raise ValueError("PhysFusion configs must enable training.label_standardization")
    label_stats = compute_train_label_stats(train_dataset)

    splits = {
        split: load_or_create_split_cache(first_config, split, label_stats, args.cache_dir)
        for split in ("train", "val", "test")
    }
    print(
        "Mean predictor check: "
        f"test AVG MAE={mae_metrics(splits['test']['labels'], np.broadcast_to(label_stats['mean'], splits['test']['labels'].shape))['AVG_MAE']:.4f} "
        f"(expected approximately {EXPECTED_MEAN_PREDICTOR_AVG_MAE:.4f})"
    )

    results: Dict[str, Any] = {
        "seeds": list(SEEDS),
        "descriptor_count": EXPECTED_DESCRIPTOR_COUNT,
        "selection_metric": "validation segment-level AVG_MAE in mmHg",
        "fit_policy": "selected hyperparameters refit on training split only",
        "label_standardization": {
            "source": "training split only",
            "mean": np.asarray(label_stats["mean"]).tolist(),
            "std": np.asarray(label_stats["std"]).tolist(),
        },
        "mean_predictor_expected_test_avg_mae": EXPECTED_MEAN_PREDICTOR_AVG_MAE,
        "per_seed": {},
    }
    for seed in SEEDS:
        print(f"\n=== Descriptor-only models, seed {seed} ===")
        results["per_seed"][str(seed)] = run_seed(
            seed,
            configs[seed],
            splits,
            label_stats,
        )
    results["summary"] = summarize_results(results["per_seed"])

    print("\nTest MAE mean +/- sample SD across seeds")
    for model_name, model_summary in results["summary"].items():
        values = " | ".join(
            f"{metric.replace('_MAE', '')}: {model_summary[metric]['mean']:.4f} +/- "
            f"{model_summary[metric]['sd']:.4f}"
            for metric in ("SBP_MAE", "DBP_MAE", "AVG_MAE")
        )
        print(f"{model_name}: {values}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"\nSaved results: {args.output}")


if __name__ == "__main__":
    main()
