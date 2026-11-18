from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import inspect
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt

from captum.attr import IntegratedGradients

from datasets.mimic_bp_dataset import MIMICBPDataset
import models

CHANNEL_NAMES = ["ECG", "PPG", "dPPG", "ddPPG"]


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_nested(cfg, keys, default=None):
    cur = cfg
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def print_config_summary(cfg):
    print("\n=== Config summary ===")
    print("Top-level keys:", list(cfg.keys()) if isinstance(cfg, dict) else "Not dict")
    print("model:", cfg.get("model", None))
    print("data:", cfg.get("data", None))
    print("dataset:", cfg.get("dataset", None))
    print("======================\n")


def create_model_from_config(cfg, model_name_override=None):
    """
    Robust model creation for your repo.
    Internally, model may still be named resnet1d_physfusion.
    Paper-facing name is PFF-ResNet.
    """

    model_name = (
        model_name_override
        or get_nested(cfg, ["model", "name"])
        or get_nested(cfg, ["model", "model_name"])
        or cfg.get("model_name")
        or cfg.get("architecture")
        or "resnet1d_physfusion"
    )

    model_cfg = cfg.get("model", {})
    if not isinstance(model_cfg, dict):
        model_cfg = {}

    # Remove name-like keys before passing as kwargs
    model_kwargs = dict(model_cfg)
    for k in ["name", "model_name", "architecture", "type"]:
        model_kwargs.pop(k, None)

    print(f"Creating model internally as: {model_name}")
    print("Model kwargs from config:", model_kwargs)

    # 1) Try repo factory functions
    for fn_name in ["get_model", "create_model", "build_model", "make_model"]:
        if hasattr(models, fn_name):
            fn = getattr(models, fn_name)
            print(f"Trying models.{fn_name} ...")

            try:
                return fn(model_name, **model_kwargs)
            except Exception as e:
                print(f"  Failed pattern fn(model_name, **kwargs): {type(e).__name__}: {e}")

            try:
                return fn(model_name)
            except Exception as e:
                print(f"  Failed pattern fn(model_name): {type(e).__name__}: {e}")

            try:
                return fn(cfg)
            except Exception as e:
                print(f"  Failed pattern fn(cfg): {type(e).__name__}: {e}")

    # 2) Try if model name is directly available in models
    if hasattr(models, model_name):
        obj = getattr(models, model_name)
        print(f"Trying direct models.{model_name} ...")

        try:
            return obj(**model_kwargs)
        except Exception as e:
            print(f"  Failed direct class kwargs: {type(e).__name__}: {e}")

        try:
            return obj()
        except Exception as e:
            print(f"  Failed direct class empty: {type(e).__name__}: {e}")

    # 3) Try direct import of known class
    try:
        from models.resnet1d import ResNet1DPhysFusion
        cls = ResNet1DPhysFusion
        print("Trying models.resnet1d.ResNet1DPhysFusion ...")

        possible_kwargs_list = [
            model_kwargs,
            {"in_channels": 4, "num_outputs": 2, **model_kwargs},
            {"input_channels": 4, "num_outputs": 2, **model_kwargs},
            {"input_channels": 4, "out_dim": 2, **model_kwargs},
            {"in_channels": 4, "out_dim": 2, **model_kwargs},
            {"num_channels": 4, "num_outputs": 2, **model_kwargs},
            {},
        ]

        for kwargs in possible_kwargs_list:
            try:
                print("  Trying kwargs:", kwargs)
                return cls(**kwargs)
            except Exception as e:
                print(f"  Failed: {type(e).__name__}: {e}")

    except Exception as e:
        print("Could not import ResNet1DPhysFusion:", type(e).__name__, e)

    print("\nAvailable names in models:")
    print([x for x in dir(models) if not x.startswith("_")])

    raise RuntimeError(
        "Could not create PFF-ResNet model automatically. "
        "Send me this full error output plus models/__init__.py."
    )


def load_checkpoint(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model" in ckpt:
            state = ckpt["model"]
        elif "net" in ckpt:
            state = ckpt["net"]
        else:
            state = ckpt
    else:
        raise RuntimeError("Unsupported checkpoint format.")

    clean_state = {}

    for k, v in state.items():
        nk = k

        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("model."):
            nk = nk[len("model."):]

        clean_state[nk] = v

    missing, unexpected = model.load_state_dict(clean_state, strict=False)

    print("\n=== Checkpoint loading ===")
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    if len(missing) > 0:
        print("First missing keys:", missing[:10])

    if len(unexpected) > 0:
        print("First unexpected keys:", unexpected[:10])

    print("==========================\n")

    return model


def create_test_dataset(cfg):
    data_dir = (
        get_nested(cfg, ["data", "data_dir"])
        or get_nested(cfg, ["dataset", "data_dir"])
        or cfg.get("data_dir")
        or os.environ.get("MIMIC_BP_DIR", "../../datasets/mimic_bp")
    )

    sampling_rate = (
        get_nested(cfg, ["data", "sampling_rate"])
        or get_nested(cfg, ["dataset", "sampling_rate"])
        or cfg.get("sampling_rate")
        or 125
    )

    print("Loading test dataset")
    print("data_dir:", data_dir)
    print("sampling_rate:", sampling_rate)
    print("include_physio_features: True")

    constructor_attempts = [
        dict(data_dir=data_dir, split="test", sampling_rate=sampling_rate, include_physio_features=True),
        dict(root_dir=data_dir, split="test", sampling_rate=sampling_rate, include_physio_features=True),
        dict(data_dir=data_dir, split="test", include_physio_features=True),
        dict(root_dir=data_dir, split="test", include_physio_features=True),
    ]

    last_error = None

    for kwargs in constructor_attempts:
        try:
            print("Trying dataset kwargs:", kwargs)
            ds = MIMICBPDataset(**kwargs)
            print("Dataset loaded successfully.")
            return ds
        except Exception as e:
            print("Failed:", type(e).__name__, e)
            last_error = e

    # Positional fallback
    try:
        ds = MIMICBPDataset(data_dir, split="test", include_physio_features=True)
        print("Dataset loaded successfully with positional fallback.")
        return ds
    except Exception as e:
        last_error = e

    raise RuntimeError(f"Could not load test dataset. Last error: {last_error}")


def move_meta_to_device(meta, device):
    new_meta = {}

    for k, v in meta.items():
        if torch.is_tensor(v):
            new_meta[k] = v.to(device)
        elif isinstance(v, np.ndarray):
            new_meta[k] = torch.tensor(v).to(device)
        elif isinstance(v, (float, int)):
            new_meta[k] = torch.tensor(v).to(device)
        else:
            new_meta[k] = v

    return new_meta


def forward_pff_resnet(model, x, meta, target_index):
    """
    target_index:
    0 = SBP
    1 = DBP
    """

    needs_physio = getattr(model, "requires_physio_features", False)

    # PFF-ResNet/PhysFusion requires physio features even if attribute is absent
    if needs_physio or "physio_features" in meta:
        physio = meta["physio_features"]

        if not torch.is_tensor(physio):
            physio = torch.tensor(physio)

        physio = physio.to(x.device).float()

        if physio.ndim == 1:
            physio = physio.unsqueeze(0)

        physio = torch.nan_to_num(physio, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            out = model(x, physio)
        except TypeError:
            out = model(x)
    else:
        out = model(x)

    if isinstance(out, tuple):
        out = out[0]

    return out[:, target_index]


def compute_ig(model, x, meta, target_index, device, n_steps):
    model.eval()

    if not torch.is_tensor(x):
        x = torch.tensor(x)

    x = x.unsqueeze(0).to(device).float()
    x.requires_grad_(True)

    meta = move_meta_to_device(meta, device)

    baseline = torch.zeros_like(x)

    def wrapped_forward(inp):
        return forward_pff_resnet(model, inp, meta, target_index)

    ig = IntegratedGradients(wrapped_forward)

    attr = ig.attribute(
        x,
        baselines=baseline,
        n_steps=n_steps,
        internal_batch_size=None,
    )

    x_np = x.detach().cpu().squeeze(0).numpy()
    attr_np = attr.detach().cpu().squeeze(0).numpy()

    return x_np, attr_np


def plot_temporal_ig(x_np, attr_np, target_name, sample_index, outdir, sampling_rate=125):
    """
    Saves temporal IG plots for ECG, PPG, dPPG, ddPPG.
    """

    T = x_np.shape[1]
    time = np.arange(T) / float(sampling_rate)

    # normalize signals only for visualization
    sig = x_np.copy()
    sig = (sig - sig.mean(axis=1, keepdims=True)) / (sig.std(axis=1, keepdims=True) + 1e-8)

    attr_abs = np.abs(attr_np)
    attr_norm = attr_abs / (attr_abs.max() + 1e-8)

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

    for ch in range(4):
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


def plot_channel_summary(scores, target_name, outdir):
    scores = np.array(scores, dtype=float)
    percentages = 100.0 * scores / (scores.sum() + 1e-8)

    plt.figure(figsize=(6, 4))
    plt.bar(CHANNEL_NAMES, percentages)
    plt.ylabel("Relative attribution (%)")
    plt.title(f"PFF-ResNet channel-wise IG attribution for {target_name}")

    for i, v in enumerate(percentages):
        plt.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)

    plt.ylim(0, max(percentages) + 8)

    png = outdir / f"ig_channel_summary_pff_resnet_{target_name.lower()}.png"
    pdf = outdir / f"ig_channel_summary_pff_resnet_{target_name.lower()}.pdf"

    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()

    print("Saved:", png)
    print("Saved:", pdf)

    return percentages


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="config_physfusion_seed2026.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/physfusion_seed2026/best_resnet1d_physfusion.pt")
    parser.add_argument("--model_name", default="resnet1d_physfusion")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--n_steps", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--outdir", default="figures/ig")

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    print("Using device:", device)
    print("Config:", args.config)
    print("Checkpoint:", args.checkpoint)
    print("Paper-facing model name: PFF-ResNet")
    print("Internal model name:", args.model_name)

    cfg = load_yaml(args.config)
    print_config_summary(cfg)

    model = create_model_from_config(cfg, model_name_override=args.model_name)
    model = load_checkpoint(model, args.checkpoint, device)
    model.to(device)
    model.eval()

    ds = create_test_dataset(cfg)
    print("Test dataset size:", len(ds))

    # ------------------------------------------------------------
    # Representative temporal plots
    # ------------------------------------------------------------
    item = ds[args.sample_index]

    if len(item) != 3:
        raise RuntimeError("Expected dataset item format: (x, y, meta)")

    x, y, meta = item

    if not isinstance(meta, dict):
        raise RuntimeError("Expected meta dictionary with physio_features.")

    print("\nSelected sample index:", args.sample_index)
    print("x shape:", x.shape if hasattr(x, "shape") else type(x))
    print("y:", y)
    print("meta keys:", list(meta.keys()))

    for target_index, target_name in [(0, "SBP"), (1, "DBP")]:
        x_np, attr_np = compute_ig(
            model=model,
            x=x,
            meta=meta,
            target_index=target_index,
            device=device,
            n_steps=args.n_steps,
        )

        plot_temporal_ig(
            x_np=x_np,
            attr_np=attr_np,
            target_name=target_name,
            sample_index=args.sample_index,
            outdir=outdir,
            sampling_rate=125,
        )

    # ------------------------------------------------------------
    # Channel-wise summary over multiple samples
    # ------------------------------------------------------------
    all_scores = {"SBP": [], "DBP": []}

    max_samples = min(args.num_samples, len(ds))
    print(f"\nComputing channel summary over {max_samples} samples...")

    for idx in range(max_samples):
        item = ds[idx]

        if len(item) != 3:
            raise RuntimeError("Expected dataset item format: (x, y, meta)")

        x, y, meta = item

        for target_index, target_name in [(0, "SBP"), (1, "DBP")]:
            _, attr_np = compute_ig(
                model=model,
                x=x,
                meta=meta,
                target_index=target_index,
                device=device,
                n_steps=args.n_steps,
            )

            channel_score = np.mean(np.abs(attr_np), axis=1)
            all_scores[target_name].append(channel_score)

        print(f"Processed sample {idx + 1}/{max_samples}")

    summary_lines = []

    for target_name in ["SBP", "DBP"]:
        scores = np.stack(all_scores[target_name], axis=0).mean(axis=0)
        percentages = plot_channel_summary(scores, target_name, outdir)

        summary_lines.append(f"{target_name} channel attribution percentages:")
        for ch, pct in zip(CHANNEL_NAMES, percentages):
            summary_lines.append(f"{ch}: {pct:.2f}%")
        summary_lines.append("")

    summary_text = "\n".join(summary_lines)

    summary_path = outdir / "ig_channel_attribution_summary_pff_resnet.txt"
    summary_path.write_text(summary_text)

    print("\n=== IG channel attribution summary ===")
    print(summary_text)
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
