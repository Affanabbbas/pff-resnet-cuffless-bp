from __future__ import annotations

from typing import Dict

import numpy as np


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if len(a) < 2:
        return float("nan")
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def compute_bp_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    y_true/y_pred shape: [N, 2], columns [SBP, DBP]
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.ndim != 2 or y_true.shape[1] != 2:
        raise ValueError(f"y_true must be [N,2], got {y_true.shape}")
    if y_pred.shape != y_true.shape:
        raise ValueError(f"y_pred shape {y_pred.shape} must match y_true {y_true.shape}")

    err = y_pred - y_true
    out = {}

    names = ["SBP", "DBP"]
    for i, name in enumerate(names):
        e = err[:, i]
        t = y_true[:, i]
        p = y_pred[:, i]
        out[f"{name}_MAE"] = float(np.mean(np.abs(e)))
        out[f"{name}_RMSE"] = float(np.sqrt(np.mean(e ** 2)))
        out[f"{name}_ME"] = float(np.mean(e))
        out[f"{name}_STD"] = float(np.std(e, ddof=1)) if len(e) > 1 else 0.0
        out[f"{name}_Pearson"] = _pearson(t, p)

    out["AVG_MAE"] = float((out["SBP_MAE"] + out["DBP_MAE"]) / 2.0)
    out["AVG_RMSE"] = float((out["SBP_RMSE"] + out["DBP_RMSE"]) / 2.0)

    # Common AAMI-style sanity indicators. This is not a replacement for a formal standard evaluation.
    out["SBP_AAMI_ME_OK"] = float(abs(out["SBP_ME"]) <= 5.0)
    out["DBP_AAMI_ME_OK"] = float(abs(out["DBP_ME"]) <= 5.0)
    out["SBP_AAMI_STD_OK"] = float(out["SBP_STD"] <= 8.0)
    out["DBP_AAMI_STD_OK"] = float(out["DBP_STD"] <= 8.0)
    return out


def format_metrics(metrics: Dict[str, float]) -> str:
    keys = [
        "SBP_MAE", "SBP_RMSE", "SBP_Pearson",
        "DBP_MAE", "DBP_RMSE", "DBP_Pearson",
        "AVG_MAE",
    ]
    return " | ".join([f"{k}={metrics[k]:.4f}" for k in keys if k in metrics])
