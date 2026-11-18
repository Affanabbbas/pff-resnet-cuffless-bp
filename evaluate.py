from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.mimic_bp_dataset import MIMICBPDataset
from models import build_model
from utils.metrics import compute_bp_metrics, format_metrics
from utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--split_dir", type=str, default=None)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out_csv", type=str, default="outputs/test_predictions.csv")
    p.add_argument("--metrics_json", type=str, default="outputs/test_metrics.json")
    p.add_argument("--subject_csv", type=str, default="outputs/test_subject_predictions.csv")
    p.add_argument("--no_subject_metrics", action="store_true")
    return p.parse_args()


def ensure_config_defaults(cfg: Dict) -> Dict:
    cfg.setdefault("signals", {})
    cfg.setdefault("training", {})
    cfg.setdefault("model", {})

    signals = cfg["signals"]
    signals.setdefault("use_derivatives", False)
    signals.setdefault("segment_seconds", 30)
    signals.setdefault("sampling_rate", 125)
    signals.setdefault("expected_segments_per_subject", 30)
    signals.setdefault("normalization", "global_train")
    signals.setdefault("max_cache_subjects", 16)
    signals.setdefault("preprocess", False)
    signals.setdefault("filter_signals", True)
    signals.setdefault("quality_filter", False)
    signals.setdefault("min_ppg_peak_count", 10)
    signals.setdefault("include_physio_features", False)

    training = cfg["training"]
    training.setdefault("seed", 42)
    training.setdefault("label_standardization", False)
    training.setdefault("loss", "huber")

    # Needed when evaluating checkpoints trained with Pearson / STD losses
    training.setdefault("pearson_alpha", 0.10)
    training.setdefault("std_alpha", 0.05)

    model = cfg["model"]
    model.setdefault("base_channels", 64)
    model.setdefault("d_model", 128)
    model.setdefault("mamba_layers", 3)
    model.setdefault("mamba_expansion", 2)
    model.setdefault("attention_heads", 4)
    model.setdefault("dropout", 0.15)

    return cfg


# =========================
# LOSS: MAE + PEARSON
# =========================
class MAEPearsonLoss(nn.Module):
    """
    Loss = MAE + alpha * (1 - Pearson correlation)

    Used only to compute checkpoint loss during evaluation.
    Final reported MAE/RMSE/Pearson metrics are still computed separately
    in real mmHg units.
    """

    def __init__(self, alpha: float = 0.10, eps: float = 1e-8):
        super().__init__()
        self.alpha = float(alpha)
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mae = torch.mean(torch.abs(pred - target))

        pred_c = pred - pred.mean(dim=0, keepdim=True)
        target_c = target - target.mean(dim=0, keepdim=True)

        numerator = torch.sum(pred_c * target_c, dim=0)
        denominator = torch.sqrt(torch.sum(pred_c ** 2, dim=0) + self.eps) * torch.sqrt(
            torch.sum(target_c ** 2, dim=0) + self.eps
        )

        corr = numerator / denominator
        corr_loss = 1.0 - corr.mean()

        return mae + self.alpha * corr_loss


# =========================
# LOSS: MAE + PEARSON + STD
# =========================
class MAEPearsonStdLoss(nn.Module):
    """
    Loss = MAE + pearson_alpha * Pearson loss + std_alpha * STD matching loss

    MAE keeps prediction close to true BP.
    Pearson helps preserve high/low BP ranking.
    STD matching reduces mean-regression by encouraging prediction spread.

    Used only to compute checkpoint loss during evaluation.
    Final reported MAE/RMSE/Pearson metrics are still computed separately
    in real mmHg units.
    """

    def __init__(self, pearson_alpha: float = 0.03, std_alpha: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.pearson_alpha = float(pearson_alpha)
        self.std_alpha = float(std_alpha)
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mae = torch.mean(torch.abs(pred - target))

        # Pearson term, computed per output dimension: SBP and DBP
        pred_c = pred - pred.mean(dim=0, keepdim=True)
        target_c = target - target.mean(dim=0, keepdim=True)

        pearson = (pred_c * target_c).sum(dim=0) / (
            torch.sqrt((pred_c ** 2).sum(dim=0) + self.eps)
            * torch.sqrt((target_c ** 2).sum(dim=0) + self.eps)
        )

        pearson_loss = torch.mean(1.0 - pearson)

        # STD-matching term, computed per output dimension: SBP and DBP
        pred_std = torch.std(pred, dim=0, unbiased=False)
        target_std = torch.std(target, dim=0, unbiased=False)

        std_loss = torch.mean(torch.abs(pred_std - target_std))

        return mae + self.pearson_alpha * pearson_loss + self.std_alpha * std_loss


def make_loss(name: str, pearson_alpha: float = 0.10, std_alpha: float = 0.05):
    name = str(name).lower()

    if name == "huber":
        return nn.SmoothL1Loss(beta=5.0)

    if name == "mse":
        return nn.MSELoss()

    if name == "mae":
        return nn.L1Loss()

    if name == "mae_pearson":
        return MAEPearsonLoss(alpha=pearson_alpha)

    if name == "mae_pearson_std":
        return MAEPearsonStdLoss(
            pearson_alpha=pearson_alpha,
            std_alpha=std_alpha,
        )

    raise ValueError(f"Unknown loss: {name}")



def forward_model(model: nn.Module, x: torch.Tensor, meta: dict, device: torch.device) -> torch.Tensor:
    """
    Normal models use model(x).
    Physio-fusion models use model(x, physio_features).
    """
    if getattr(model, "requires_physio_features", False):
        if "physio_features" not in meta:
            raise KeyError("Model requires physio_features, but meta['physio_features'] is missing.")

        physio = meta["physio_features"].to(device).float()
        physio = torch.nan_to_num(physio, nan=0.0, posinf=0.0, neginf=0.0)

        return model(x, physio)

    return model(x)

def _stats_from_checkpoint(ckpt: Dict, key: str) -> Optional[Dict[str, np.ndarray]]:
    stats = ckpt.get(key, None)

    if stats is None and key == "signal_stats":
        stats = ckpt.get("stats", None)

    if stats is None:
        return None

    return {k: np.asarray(v, dtype=np.float32) for k, v in stats.items()}


def invert_label_standardization(
    arr: np.ndarray,
    label_stats: Optional[Dict[str, np.ndarray]],
) -> np.ndarray:
    if label_stats is None:
        return arr

    mean = np.asarray(label_stats["mean"], dtype=np.float32).reshape(1, 2)
    std = np.asarray(label_stats["std"], dtype=np.float32).reshape(1, 2)

    return arr * (std + 1e-8) + mean


def make_dataset_kwargs(
    cfg: Dict,
    signal_stats,
    label_stats,
    standardize_labels: bool,
    split_dir,
    verbose: bool,
) -> Dict:
    signals = cfg["signals"]

    return {
        "split_dir": split_dir,
        "use_derivatives": bool(signals["use_derivatives"]),
        "normalization": signals["normalization"],
        "stats": signal_stats,
        "sampling_rate": int(signals["sampling_rate"]),
        "segment_seconds": int(signals["segment_seconds"]),
        "expected_segments_per_subject": int(signals["expected_segments_per_subject"]),
        "max_cache_subjects": int(signals["max_cache_subjects"]),
        "verbose": verbose,
        "preprocess": bool(signals["preprocess"]),
        "filter_signals": bool(signals["filter_signals"]),
        "quality_filter": bool(signals["quality_filter"]),
        "min_ppg_peak_count": int(signals["min_ppg_peak_count"]),
        "include_physio_features": bool(signals["include_physio_features"]),
        "standardize_labels": bool(standardize_labels),
        "label_stats": label_stats,
    }


def compute_subject_rows(rows):
    grouped = defaultdict(list)

    for row in rows:
        grouped[row["subject_id"]].append(row)

    subject_rows = []

    for subject_id, items in sorted(grouped.items()):
        true = np.asarray(
            [[r["true_sbp"], r["true_dbp"]] for r in items],
            dtype=np.float64,
        )
        pred = np.asarray(
            [[r["pred_sbp"], r["pred_dbp"]] for r in items],
            dtype=np.float64,
        )

        subject_rows.append(
            {
                "subject_id": subject_id,
                "n_segments": len(items),
                "true_sbp": float(true[:, 0].mean()),
                "true_dbp": float(true[:, 1].mean()),
                "pred_sbp": float(pred[:, 0].mean()),
                "pred_dbp": float(pred[:, 1].mean()),
                "abs_err_sbp": float(abs(pred[:, 0].mean() - true[:, 0].mean())),
                "abs_err_dbp": float(abs(pred[:, 1].mean() - true[:, 1].mean())),
            }
        )

    return subject_rows


def write_csv(path: str | Path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError(f"No rows to write for {path}")

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def main():
    args = parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ensure_config_defaults(ckpt["config"])

    set_seed(int(cfg["training"].get("seed", 42)))

    data_dir = args.data_dir or cfg["data_dir"]
    split_dir = args.split_dir or cfg.get("split_dir", None)

    signal_stats = _stats_from_checkpoint(ckpt, "signal_stats")
    label_stats = _stats_from_checkpoint(ckpt, "label_stats")

    standardize_labels = (
        bool(cfg["training"].get("label_standardization", False))
        and label_stats is not None
    )

    ds = MIMICBPDataset(
        data_dir=data_dir,
        split=args.split,
        **make_dataset_kwargs(
            cfg,
            signal_stats=signal_stats,
            label_stats=label_stats,
            standardize_labels=standardize_labels,
            split_dir=split_dir,
            verbose=True,
        ),
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA requested but not available. Using CPU.")
        device = "cpu"

    device = torch.device(device)

    model_name = str(ckpt.get("model_name", cfg["training"].get("model", "resnet1d")))

    model = build_model(
        model_name=model_name,
        in_channels=int(ckpt["in_channels"]),
        cfg=cfg,
    )

    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    loss_name = cfg["training"].get("loss", "huber")
    pearson_alpha = float(cfg["training"].get("pearson_alpha", 0.10))
    std_alpha = float(cfg["training"].get("std_alpha", 0.05))

    criterion = make_loss(
        loss_name,
        pearson_alpha=pearson_alpha,
        std_alpha=std_alpha,
    )

    y_true_all, y_pred_all = [], []
    rows = []
    total_loss = 0.0

    print(f"Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"Checkpoint best metric: {ckpt.get('best_metric', 'unknown')}={ckpt.get('best_score', 'unknown')}")
    print(f"Label standardization: {standardize_labels}")
    print(f"Loss: {loss_name}")

    if str(loss_name).lower() in {"mae_pearson", "mae_pearson_std"}:
        print(f"Pearson alpha: {pearson_alpha}")

    if str(loss_name).lower() == "mae_pearson_std":
        print(f"STD alpha: {std_alpha}")

    for x, y, meta in tqdm(loader, desc=f"evaluate:{args.split}"):
        x = x.to(device, non_blocking=True).float()
        y_dev = y.to(device, non_blocking=True).float()

        pred = forward_model(model, x, meta, device)

        loss = criterion(pred, y_dev)
        total_loss += loss.item() * x.size(0)

        y_np = y.numpy()
        pred_np = pred.detach().cpu().numpy()

        y_mmHg = invert_label_standardization(
            y_np,
            label_stats if standardize_labels else None,
        )
        pred_mmHg = invert_label_standardization(
            pred_np,
            label_stats if standardize_labels else None,
        )

        y_true_all.append(y_mmHg)
        y_pred_all.append(pred_mmHg)

        subjects = meta["subject_id"]
        segs = meta["segment_id"]

        if torch.is_tensor(segs):
            segs = segs.cpu().numpy().tolist()

        for i in range(len(subjects)):
            rows.append(
                {
                    "subject_id": subjects[i],
                    "segment_id": int(segs[i]),
                    "true_sbp": float(y_mmHg[i, 0]),
                    "true_dbp": float(y_mmHg[i, 1]),
                    "pred_sbp": float(pred_mmHg[i, 0]),
                    "pred_dbp": float(pred_mmHg[i, 1]),
                    "abs_err_sbp": float(abs(pred_mmHg[i, 0] - y_mmHg[i, 0])),
                    "abs_err_dbp": float(abs(pred_mmHg[i, 1] - y_mmHg[i, 1])),
                }
            )

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)

    metrics = compute_bp_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    metrics["n_segments"] = int(len(rows))
    metrics["n_subjects"] = int(len(set(row["subject_id"] for row in rows)))

    if not args.no_subject_metrics:
        subject_rows = compute_subject_rows(rows)

        subject_true = np.asarray(
            [[r["true_sbp"], r["true_dbp"]] for r in subject_rows],
            dtype=np.float64,
        )
        subject_pred = np.asarray(
            [[r["pred_sbp"], r["pred_dbp"]] for r in subject_rows],
            dtype=np.float64,
        )

        subject_metrics = compute_bp_metrics(subject_true, subject_pred)
        metrics["subject_level"] = subject_metrics

    else:
        subject_rows = []

    print(format_metrics(metrics))
    print(json.dumps(metrics, indent=2))

    write_csv(args.out_csv, rows)

    if not args.no_subject_metrics:
        write_csv(args.subject_csv, subject_rows)

    metrics_json = Path(args.metrics_json)
    metrics_json.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved predictions: {args.out_csv}")

    if not args.no_subject_metrics:
        print(f"Saved subject predictions: {args.subject_csv}")

    print(f"Saved metrics: {metrics_json}")


if __name__ == "__main__":
    main()
