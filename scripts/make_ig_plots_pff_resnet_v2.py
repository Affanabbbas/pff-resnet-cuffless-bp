from pathlib import Path
import os
import sys
import argparse
import inspect
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from captum.attr import IntegratedGradients
from datasets.mimic_bp_dataset import MIMICBPDataset
import models

CHANNEL_NAMES = ["ECG", "PPG", "dPPG", "ddPPG"]

PHYSIO_FEATURE_NAMES = [
    "ECG mean",
    "ECG std",
    "ECG peak-to-peak",
    "PPG mean",
    "PPG std",
    "PPG peak-to-peak",
    "Mean |dPPG|",
    "Mean |ddPPG|",
    "ECG HR",
    "PPG HR",
    "Median PTT",
    "PPG peak count",
]


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def normalize_signals(signals):
    if signals is None:
        return ["ecg", "ppg", "dppg", "ddppg"]
    if isinstance(signals, str):
        return [s.strip() for s in signals.split(",")]
    return list(signals)


def first_not_none(*values):
    for v in values:
        if v is not None:
            return v
    return None


def get_checkpoint_state(ckpt):
    if not isinstance(ckpt, dict):
        raise RuntimeError("Checkpoint is not a dictionary.")

    for key in ["model_state", "model_state_dict", "state_dict", "model", "net"]:
        if key in ckpt:
            print(f"Using checkpoint state key: {key}")
            return ckpt[key]

    print("Checkpoint keys:", ckpt.keys())
    raise RuntimeError("Could not find model state in checkpoint.")


def clean_state_dict(state):
    clean = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("model."):
            nk = nk[len("model."):]
        clean[nk] = v
    return clean


def build_pff_model(cfg, model_name):
    signals = normalize_signals(cfg.get("signals"))
    in_channels = len(signals)

    model_cfg = cfg.get("model", {})
    if not isinstance(model_cfg, dict):
        model_cfg = {}

    print("\n=== Model creation ===")
    print("Paper name: PFF-ResNet")
    print("Internal name:", model_name)
    print("Signals:", signals)
    print("Input channels:", in_channels)
    print("Model config:", model_cfg)

    if hasattr(models, "build_model"):
        try:
            print("Trying models.build_model(model_name, in_channels, model_cfg)")
            return models.build_model(model_name, in_channels, model_cfg)
        except Exception as e:
            print("Failed:", type(e).__name__, e)

        try:
            print("Trying models.build_model(model_name, in_channels, full_cfg)")
            return models.build_model(model_name, in_channels, cfg)
        except Exception as e:
            print("Failed:", type(e).__name__, e)

    from models.resnet1d import ResNet1DPhysFusion

    sig = inspect.signature(ResNet1DPhysFusion)
    allowed = set(sig.parameters.keys())

    kwargs = {}
    if "in_channels" in allowed:
        kwargs["in_channels"] = in_channels

    for k, v in model_cfg.items():
        if k in allowed:
            kwargs[k] = v

    print("Trying direct ResNet1DPhysFusion with kwargs:", kwargs)
    return ResNet1DPhysFusion(**kwargs)


def load_model_weights(model, ckpt):
    state = get_checkpoint_state(ckpt)
    state = clean_state_dict(state)

    missing, unexpected = model.load_state_dict(state, strict=False)

    print("\n=== Checkpoint loading ===")
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    if len(missing) > 0:
        print("First missing:", missing[:10])
    if len(unexpected) > 0:
        print("First unexpected:", unexpected[:10])

    print("==========================\n")
    return model


def make_dataset(cfg, ckpt):
    data_dir = cfg.get("data_dir") or os.environ.get("MIMIC_BP_DIR", "../../datasets/mimic_bp")
    split_dir = cfg.get("split_dir", None)
    signals = normalize_signals(cfg.get("signals"))

    signal_stats = first_not_none(
        ckpt.get("signal_stats") if isinstance(ckpt, dict) else None,
        ckpt.get("stats", {}).get("signal_stats") if isinstance(ckpt, dict) and isinstance(ckpt.get("stats"), dict) else None,
    )

    label_stats = first_not_none(
        ckpt.get("label_stats") if isinstance(ckpt, dict) else None,
        ckpt.get("stats", {}).get("label_stats") if isinstance(ckpt, dict) and isinstance(ckpt.get("stats"), dict) else None,
    )

    sig = inspect.signature(MIMICBPDataset)
    params = set(sig.parameters.keys())

    kwargs = {}

    if "data_dir" in params:
        kwargs["data_dir"] = data_dir
    elif "root_dir" in params:
        kwargs["root_dir"] = data_dir

    if "split" in params:
        kwargs["split"] = "test"

    if "split_dir" in params and split_dir is not None:
        kwargs["split_dir"] = split_dir

    if "signals" in params:
        kwargs["signals"] = signals

    if "include_physio_features" in params:
        kwargs["include_physio_features"] = True

    if "sampling_rate" in params:
        kwargs["sampling_rate"] = 125

    if "normalization" in params:
        kwargs["normalization"] = "global_train"

    if "train_stats" in params:
        kwargs["train_stats"] = signal_stats

    if "signal_stats" in params:
        kwargs["signal_stats"] = signal_stats

    if "label_stats" in params:
        kwargs["label_stats"] = label_stats

    print("\n=== Dataset creation ===")
    print("Dataset kwargs:")
    for k, v in kwargs.items():
        if k in ["train_stats", "signal_stats", "label_stats"]:
            print(f"{k}: loaded")
        else:
            print(f"{k}: {v}")

    ds = MIMICBPDataset(**kwargs)
    print("Dataset size:", len(ds))
    print("========================\n")
    return ds


def to_tensor_2d_physio(v, device):
    if not torch.is_tensor(v):
        v = torch.tensor(v)
    v = v.float().to(device)
    if v.ndim == 1:
        v = v.unsqueeze(0)
    return torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


def get_sample(ds, idx, device):
    item = ds[idx]
    if len(item) != 3:
        raise RuntimeError("Expected dataset item as (x, y, meta).")

    x, y, meta = item

    if not torch.is_tensor(x):
        x = torch.tensor(x)

    x = x.float().unsqueeze(0).to(device)
    x.requires_grad_(True)

    if not isinstance(meta, dict):
        raise RuntimeError("Expected meta dictionary.")

    if "physio_features" not in meta:
        raise RuntimeError("meta does not contain physio_features.")

    physio = to_tensor_2d_physio(meta["physio_features"], device)
    physio.requires_grad_(True)

    return x, physio, y, meta


def model_forward(model, x, physio, target_index):
    try:
        out = model(x, physio)
    except TypeError:
        out = model(x)

    if isinstance(out, tuple):
        out = out[0]

    return out[:, target_index]


def compute_ig(model, x, physio, target_index, n_steps):
    baseline_x = torch.zeros_like(x)
    baseline_physio = torch.zeros_like(physio)

    def wrapped_forward(x_in, physio_in):
        return model_forward(model, x_in, physio_in, target_index)

    ig = IntegratedGradients(wrapped_forward)

    attr_x, attr_physio = ig.attribute(
        inputs=(x, physio),
        baselines=(baseline_x, baseline_physio),
        n_steps=n_steps,
    )

    return (
        attr_x.detach().cpu().squeeze(0).numpy(),
        attr_physio.detach().cpu().squeeze(0).numpy(),
    )


def plot_temporal(x_np, attr_x, target_name, sample_index, outdir):
    T = x_np.shape[1]
    time = np.arange(T) / 125.0

    sig = x_np.copy()
    sig = (sig - sig.mean(axis=1, keepdims=True)) / (sig.std(axis=1, keepdims=True) + 1e-8)

    attr_abs = np.abs(attr_x)
    attr_norm = attr_abs / (attr_abs.max() + 1e-8)

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

    for ch in range(min(4, sig.shape[0])):
        threshold = np.percentile(attr_norm[ch], 85)

        axes[ch].plot(time, sig[ch], linewidth=0.8)
        axes[ch].fill_between(
            time,
            sig[ch].min(),
            sig[ch].max(),
            where=attr_norm[ch] >= threshold,
            alpha=0.25,
        )
        axes[ch].set_ylabel(CHANNEL_NAMES[ch])
        axes[ch].grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"PFF-ResNet Integrated Gradients temporal attribution for {target_name}")
    fig.tight_layout()

    png = outdir / f"ig_temporal_pff_resnet_{target_name.lower()}_sample{sample_index}.png"
    pdf = outdir / f"ig_temporal_pff_resnet_{target_name.lower()}_sample{sample_index}.pdf"

    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", png)
    print("Saved:", pdf)


def plot_bar(names, values, ylabel, title, out_png, out_pdf):
    values = np.array(values, dtype=float)
    percentages = 100.0 * values / (values.sum() + 1e-8)

    plt.figure(figsize=(8, 4.5))
    plt.bar(names, percentages)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=30, ha="right")

    for i, v in enumerate(percentages):
        plt.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=8)

    plt.ylim(0, max(percentages) + 8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    print("Saved:", out_png)
    print("Saved:", out_pdf)

    return percentages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_physfusion_seed2026.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/physfusion_seed2026/best_resnet1d_physfusion.pt")
    parser.add_argument("--model_name", default="resnet1d_physfusion")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--outdir", default="figures/ig")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    print("Using device:", device)
    print("Config:", args.config)
    print("Checkpoint:", args.checkpoint)

    cfg = load_yaml(args.config)
    ckpt = torch.load(args.checkpoint, map_location=device)

    model = build_pff_model(cfg, args.model_name)
    model = load_model_weights(model, ckpt)
    model.to(device)
    model.eval()

    ds = make_dataset(cfg, ckpt)

    # representative temporal IG plot
    x, physio, y, meta = get_sample(ds, args.sample_index, device)

    print("Selected sample:", args.sample_index)
    print("x shape:", tuple(x.shape))
    print("physio shape:", tuple(physio.shape))
    print("y:", y)

    x_np = x.detach().cpu().squeeze(0).numpy()

    all_channel_scores = {"SBP": [], "DBP": []}
    all_physio_scores = {"SBP": [], "DBP": []}

    for target_index, target_name in [(0, "SBP"), (1, "DBP")]:
        attr_x, attr_physio = compute_ig(model, x, physio, target_index, args.n_steps)
        plot_temporal(x_np, attr_x, target_name, args.sample_index, outdir)

    # summary across samples
    max_samples = min(args.num_samples, len(ds))
    print(f"\nComputing IG summary over {max_samples} samples...")

    for idx in range(max_samples):
        x, physio, y, meta = get_sample(ds, idx, device)

        for target_index, target_name in [(0, "SBP"), (1, "DBP")]:
            attr_x, attr_physio = compute_ig(model, x, physio, target_index, args.n_steps)

            channel_scores = np.mean(np.abs(attr_x), axis=1)
            physio_scores = np.abs(attr_physio)

            all_channel_scores[target_name].append(channel_scores)
            all_physio_scores[target_name].append(physio_scores)

        print(f"Processed {idx + 1}/{max_samples}")

    summary_lines = []

    for target_name in ["SBP", "DBP"]:
        channel_mean = np.stack(all_channel_scores[target_name], axis=0).mean(axis=0)
        physio_mean = np.stack(all_physio_scores[target_name], axis=0).mean(axis=0)

        channel_pct = plot_bar(
            CHANNEL_NAMES[:len(channel_mean)],
            channel_mean,
            "Relative attribution (%)",
            f"PFF-ResNet channel-wise IG attribution for {target_name}",
            outdir / f"ig_channel_summary_pff_resnet_{target_name.lower()}.png",
            outdir / f"ig_channel_summary_pff_resnet_{target_name.lower()}.pdf",
        )

        physio_pct = plot_bar(
            PHYSIO_FEATURE_NAMES[:len(physio_mean)],
            physio_mean,
            "Relative attribution (%)",
            f"PFF-ResNet physiological-feature IG attribution for {target_name}",
            outdir / f"ig_physio_feature_summary_pff_resnet_{target_name.lower()}.png",
            outdir / f"ig_physio_feature_summary_pff_resnet_{target_name.lower()}.pdf",
        )

        summary_lines.append(f"{target_name} waveform-channel attribution:")
        for name, pct in zip(CHANNEL_NAMES, channel_pct):
            summary_lines.append(f"{name}: {pct:.2f}%")
        summary_lines.append("")

        summary_lines.append(f"{target_name} physiological-feature attribution:")
        for name, pct in zip(PHYSIO_FEATURE_NAMES, physio_pct):
            summary_lines.append(f"{name}: {pct:.2f}%")
        summary_lines.append("")

    summary_path = outdir / "ig_summary_pff_resnet.txt"
    summary_path.write_text("\n".join(summary_lines))

    print("\n=== IG summary ===")
    print("\n".join(summary_lines))
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
