from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from utils.reproducibility import set_global_seed, seed_worker, make_generator
from tqdm import tqdm

from datasets.mimic_bp_dataset import (
    MIMICBPDataset,
    compute_train_label_stats,
    compute_train_signal_stats,
)
from models import build_model, count_parameters
from utils.leakage_check import check_subject_leakage
from utils.metrics import compute_bp_metrics, format_metrics
from utils.seed import set_seed


def load_config(path: str | None):
    if path is None:
        path = "config.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--split_dir", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--model", type=str, default=None, choices=["resnet1d",    	"cnn_mamba_attention","resnet1d_dualhead_sharedattn",
        "resnet1d_dualhead_avg",
        "resnet1d_dualhead_sepattn",])

    # UPDATED: added mae_pearson and mae_pearson_std
    p.add_argument(
        "--loss",
        type=str,
        default=None,
        choices=["huber", "mse", "mae", "mae_pearson", "mae_pearson_std"],
    )

    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--use_derivatives", action="store_true")
    p.add_argument("--preprocess", action="store_true")
    p.add_argument("--quality_filter", action="store_true")
    p.add_argument("--label_standardization", action="store_true")
    p.add_argument("--normalization", type=str, default=None, choices=["none", "global_train", "per_segment_zscore"])
    p.add_argument("--best_metric", type=str, default=None)
    p.add_argument("--save_dir", type=str, default=None)

    # OPTIONAL: control Pearson / STD loss strengths from command line
    p.add_argument("--pearson_alpha", type=float, default=None)
    p.add_argument("--std_alpha", type=float, default=None)

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

    training.setdefault("model", "resnet1d")
    training.setdefault("batch_size", 64)
    training.setdefault("epochs", 50)
    training.setdefault("lr", 3e-4)
    training.setdefault("weight_decay", 1e-4)
    training.setdefault("loss", "huber")
    training.setdefault("early_stopping_patience", 8)
    training.setdefault("num_workers", 4)
    training.setdefault("seed", 2026)
    training.setdefault("device", "cuda")
    training.setdefault("save_dir", "checkpoints")
    training.setdefault("log_csv", "outputs/train_log.csv")
    training.setdefault("label_standardization", False)
    training.setdefault("best_metric", "AVG_MAE")
    training.setdefault("scheduler_metric", "loss")

    # Loss weights for MAE + Pearson / STD losses
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


def apply_cli_overrides(cfg: Dict, args) -> Dict:
    if args.data_dir is not None:
        cfg["data_dir"] = args.data_dir
    if args.split_dir is not None:
        cfg["split_dir"] = args.split_dir
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.model is not None:
        cfg["training"]["model"] = args.model
    if args.loss is not None:
        cfg["training"]["loss"] = args.loss
    if args.num_workers is not None:
        cfg["training"]["num_workers"] = args.num_workers
    if args.device is not None:
        cfg["training"]["device"] = args.device
    if args.normalization is not None:
        cfg["signals"]["normalization"] = args.normalization
    if args.best_metric is not None:
        cfg["training"]["best_metric"] = args.best_metric
    if args.save_dir is not None:
        cfg["training"]["save_dir"] = args.save_dir
    if args.pearson_alpha is not None:
        cfg["training"]["pearson_alpha"] = args.pearson_alpha
    if args.std_alpha is not None:
        cfg["training"]["std_alpha"] = args.std_alpha
    if args.use_derivatives:
        cfg["signals"]["use_derivatives"] = True
    if args.preprocess:
        cfg["signals"]["preprocess"] = True
    if args.quality_filter:
        cfg["signals"]["quality_filter"] = True
    if args.label_standardization:
        cfg["training"]["label_standardization"] = True
    return cfg


# =========================
# NEW LOSS: MAE + PEARSON
# =========================
class MAEPearsonLoss(nn.Module):
    """
    Loss = MAE + alpha * (1 - Pearson correlation)

    This helps when the model predicts close to the mean BP and does not capture
    enough prediction variation.
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


class MAEPearsonStdLoss(nn.Module):
    """
    Loss = MAE + pearson_alpha * Pearson loss + std_alpha * STD matching loss

    MAE keeps prediction close to true BP.
    Pearson helps preserve high/low BP ranking.
    STD matching reduces mean-regression by encouraging prediction spread.
    """

    def __init__(self, pearson_alpha=0.03, std_alpha=0.05, eps=1e-8):
        super().__init__()
        self.pearson_alpha = float(pearson_alpha)
        self.std_alpha = float(std_alpha)
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mae = torch.mean(torch.abs(pred - target))

        # Pearson term
        pred_c = pred - pred.mean(dim=0, keepdim=True)
        target_c = target - target.mean(dim=0, keepdim=True)

        pearson = (pred_c * target_c).sum(dim=0) / (
            torch.sqrt((pred_c ** 2).sum(dim=0) + self.eps)
            * torch.sqrt((target_c ** 2).sum(dim=0) + self.eps)
        )

        pearson_loss = torch.mean(1.0 - pearson)

        # STD-matching term
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

def _to_serializable_stats(stats: Optional[Dict[str, np.ndarray]]) -> Optional[Dict[str, list]]:
    if stats is None:
        return None
    return {k: np.asarray(v).tolist() for k, v in stats.items()}


def invert_label_standardization(arr: np.ndarray, label_stats: Optional[Dict[str, np.ndarray]]) -> np.ndarray:
    if label_stats is None:
        return arr

    mean = np.asarray(label_stats["mean"], dtype=np.float32).reshape(1, 2)
    std = np.asarray(label_stats["std"], dtype=np.float32).reshape(1, 2)
    return arr * (std + 1e-8) + mean


def make_dataset_kwargs(
    cfg: Dict,
    stats=None,
    label_stats=None,
    standardize_labels: bool = False,
    verbose: bool = True,
) -> Dict:
    signals = cfg["signals"]

    return {
        "split_dir": cfg.get("split_dir", None),
        "use_derivatives": bool(signals["use_derivatives"]),
        "normalization": signals["normalization"],
        "stats": stats,
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


@torch.no_grad()
def run_eval(model, loader, device, criterion, label_stats=None):
    model.eval()

    y_true_all, y_pred_all = [], []
    total_loss = 0.0

    for x, y, meta in tqdm(loader, desc="eval", leave=False):
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).float()

        pred = forward_model(model, x, meta, device)
        loss = criterion(pred, y)
        total_loss += loss.item() * x.size(0)

        y_np = y.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()

        y_true_all.append(invert_label_standardization(y_np, label_stats))
        y_pred_all.append(invert_label_standardization(pred_np, label_stats))

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)

    metrics = compute_bp_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)

    return metrics


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0

    for x, y, meta in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        pred = forward_model(model, x, meta, device)
        loss = criterion(pred, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / max(len(loader.dataset), 1)


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    cfg,
    signal_stats,
    label_stats,
    best_metric,
    best_score,
    best_val_loss,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "config": cfg,
        "stats": _to_serializable_stats(signal_stats),
        "signal_stats": _to_serializable_stats(signal_stats),
        "label_stats": _to_serializable_stats(label_stats),
        "best_metric": best_metric,
        "best_score": float(best_score),
        "best_val_loss": float(best_val_loss),
        "model_name": str(cfg["training"]["model"]),
        "in_channels": int(getattr(model, "in_channels", 0)),
    }

    torch.save(checkpoint, path)


def metric_is_better(metric_name: str, score: float, best_score: float) -> bool:
    lower_is_better = {
        "loss",
        "AVG_MAE",
        "AVG_RMSE",
        "SBP_MAE",
        "DBP_MAE",
        "SBP_RMSE",
        "DBP_RMSE",
        "SBP_STD",
        "DBP_STD",
    }

    if metric_name in lower_is_better:
        return score < best_score

    return score > best_score


def initial_best_score(metric_name: str) -> float:
    lower_is_better = {
        "loss",
        "AVG_MAE",
        "AVG_RMSE",
        "SBP_MAE",
        "DBP_MAE",
        "SBP_RMSE",
        "DBP_RMSE",
        "SBP_STD",
        "DBP_STD",
    }

    return float("inf") if metric_name in lower_is_better else -float("inf")


def main():
    args = parse_args()

    cfg = ensure_config_defaults(load_config(args.config))
    cfg = apply_cli_overrides(cfg, args)

    set_seed(int(cfg["training"]["seed"]))

    data_dir = cfg["data_dir"]
    split_dir = cfg.get("split_dir", None)
    signals = cfg["signals"]
    training = cfg["training"]

    check_subject_leakage(data_dir, split_dir, raise_on_leakage=True)

    stats_dataset_kwargs = make_dataset_kwargs(
        cfg,
        stats=None,
        label_stats=None,
        standardize_labels=False,
        verbose=True,
    )
    stats_dataset_kwargs["normalization"] = "none"
    stats_dataset_kwargs["stats"] = None
    stats_dataset_kwargs["standardize_labels"] = False
    stats_dataset_kwargs["label_stats"] = None

    train_for_stats = MIMICBPDataset(
        data_dir=data_dir,
        split="train",
        **stats_dataset_kwargs,
    )

    if signals["normalization"] == "global_train":
        signal_stats = compute_train_signal_stats(train_for_stats)
    else:
        signal_stats = {
            "mean": np.zeros(train_for_stats.num_channels, dtype=np.float32),
            "std": np.ones(train_for_stats.num_channels, dtype=np.float32),
        }

    label_stats = None

    if bool(training["label_standardization"]):
        label_stats = compute_train_label_stats(train_for_stats)

    train_ds = MIMICBPDataset(
        data_dir=data_dir,
        split="train",
        **make_dataset_kwargs(
            cfg,
            stats=signal_stats,
            label_stats=label_stats,
            standardize_labels=bool(training["label_standardization"]),
            verbose=False,
        ),
    )

    val_ds = MIMICBPDataset(
        data_dir=data_dir,
        split="val",
        **make_dataset_kwargs(
            cfg,
            stats=signal_stats,
            label_stats=label_stats,
            standardize_labels=bool(training["label_standardization"]),
            verbose=True,
        ),
    )

    seed = int(training.get("seed", 2026))
    set_global_seed(seed)
    loader_generator = make_generator(seed)
    print(f"DataLoader seed: {seed}")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        num_workers=int(training["num_workers"]),
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        num_workers=int(training["num_workers"]),
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )

    device = training["device"]

    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA requested but not available. Using CPU.")
        device = "cpu"

    device = torch.device(device)

    model = build_model(
        model_name=str(training["model"]),
        in_channels=train_ds.num_channels,
        cfg=cfg,
    ).to(device)

    print(f"Model: {training['model']}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Device: {device}")
    print(f"Loss: {training['loss']}")
    print(f"Best metric: {training['best_metric']}")
    print(f"Label standardization: {bool(training['label_standardization'])}")

    if str(training["loss"]).lower() in {"mae_pearson", "mae_pearson_std"}:
        print(f"Pearson alpha: {float(training.get('pearson_alpha', 0.03))}")

    if str(training["loss"]).lower() == "mae_pearson_std":
        print(f"STD alpha: {float(training.get('std_alpha', 0.05))}")

    criterion = make_loss(
        training["loss"],
        pearson_alpha=float(training.get("pearson_alpha", 0.03)),
        std_alpha=float(training.get("std_alpha", 0.05)),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["lr"]),
        weight_decay=float(training["weight_decay"]),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    save_dir = Path(training["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = save_dir / f"best_{str(training['model']).lower()}.pt"

    log_csv = Path(training["log_csv"])
    log_csv.parent.mkdir(parents=True, exist_ok=True)

    best_metric = str(training["best_metric"])
    best_score = initial_best_score(best_metric)
    best_val_loss = float("inf")
    bad_epochs = 0
    patience = int(training["early_stopping_patience"])

    fieldnames = [
        "epoch",
        "lr",
        "train_loss",
        "val_loss",
        "SBP_MAE",
        "SBP_RMSE",
        "SBP_ME",
        "SBP_STD",
        "SBP_Pearson",
        "DBP_MAE",
        "DBP_RMSE",
        "DBP_ME",
        "DBP_STD",
        "DBP_Pearson",
        "AVG_MAE",
        "AVG_RMSE",
        "best_metric",
        "best_score",
    ]

    with open(log_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    for epoch in range(1, int(training["epochs"]) + 1):
        lr = optimizer.param_groups[0]["lr"]

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_metrics = run_eval(
            model,
            val_loader,
            device,
            criterion,
            label_stats=label_stats,
        )

        val_loss = val_metrics["loss"]
        best_val_loss = min(best_val_loss, val_loss)

        scheduler.step(val_metrics.get(str(training["scheduler_metric"]), val_loss))

        if best_metric not in val_metrics:
            raise KeyError(
                f"best_metric='{best_metric}' is not available. "
                f"Choose one of: {sorted(val_metrics.keys())}"
            )

        current_score = float(val_metrics[best_metric])

        print(
            f"Epoch {epoch:03d} | LR={lr:.6f} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | {format_metrics(val_metrics)}"
        )

        with open(log_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            row = {
                "epoch": epoch,
                "lr": lr,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_metric": best_metric,
                "best_score": current_score,
            }

            for k in fieldnames:
                if k in val_metrics:
                    row[k] = val_metrics[k]

            writer.writerow(row)

        if metric_is_better(best_metric, current_score, best_score):
            best_score = current_score
            bad_epochs = 0

            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                cfg,
                signal_stats,
                label_stats,
                best_metric,
                best_score,
                best_val_loss,
            )

            print(f"  ✓ Best model saved: {checkpoint_path} ({best_metric}={best_score:.4f})")

        else:
            bad_epochs += 1
            print(f"  no improvement: {bad_epochs}/{patience} ({best_metric}={current_score:.4f})")

        if bad_epochs >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    print("Training finished.")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Best {best_metric}: {best_score:.4f}")
    print(f"Log CSV: {log_csv}")


if __name__ == "__main__":
    main()
