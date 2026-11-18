from pathlib import Path
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from make_ig_plots_pff_resnet_v3 import (
    load_yaml,
    build_pff_model,
    load_model_weights,
    make_dataset,
    get_sample,
    compute_ig,
    CHANNEL_NAMES,
)


def one_second_bins(values, sampling_rate=125):
    values = np.asarray(values)
    n_seconds = values.shape[-1] // sampling_rate
    values = values[..., : n_seconds * sampling_rate]
    return values.reshape(*values.shape[:-1], n_seconds, sampling_rate).sum(axis=-1)


def compute_temporal_stats(attr_x, sampling_rate=125):
    attr_abs = np.abs(attr_x)
    total_time = attr_abs.sum(axis=0)
    total = total_time.sum() + 1e-12

    early_n = int(5 * sampling_rate)
    early_fraction = total_time[:early_n].sum() / total

    sec_bins = one_second_bins(total_time, sampling_rate=sampling_rate)
    peak_second = int(np.argmax(sec_bins))
    peak_fraction = float(sec_bins.max() / (sec_bins.sum() + 1e-12))

    # Higher entropy means attribution is more spread across time.
    p = sec_bins / (sec_bins.sum() + 1e-12)
    entropy = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))

    return {
        "early_0_5s_fraction": float(early_fraction),
        "peak_second": peak_second,
        "peak_1s_fraction": peak_fraction,
        "temporal_entropy": entropy,
    }


def score_candidate(stats):
    """
    We prefer samples where attribution is not only at the start,
    not dominated by one second, and reasonably spread over time.
    Ideal early fraction for 5s out of 30s is around 0.17.
    """
    early = stats["early_0_5s_fraction"]
    peak = stats["peak_1s_fraction"]
    entropy = stats["temporal_entropy"]

    score = abs(early - 0.17) + peak - 0.5 * entropy
    return float(score)


def plot_clean_temporal_ig(x_np, attr_x, target_name, sample_idx, outdir, sampling_rate=125):
    channel_names = CHANNEL_NAMES[: x_np.shape[0]]

    T = x_np.shape[1]
    time = np.arange(T) / sampling_rate

    sig = x_np.copy()
    sig = (sig - sig.mean(axis=1, keepdims=True)) / (sig.std(axis=1, keepdims=True) + 1e-8)

    attr_abs = np.abs(attr_x)
    attr_norm = attr_abs / (attr_abs.max() + 1e-12)

    total_time = attr_abs.sum(axis=0)
    total_time = total_time / (total_time.max() + 1e-12)

    fig, axes = plt.subplots(6, 1, figsize=(12, 9), sharex=True)

    for ch in range(len(channel_names)):
        axes[ch].plot(time, sig[ch], linewidth=0.8)
        axes[ch].set_ylabel(channel_names[ch])
        axes[ch].grid(True, alpha=0.25)

    # Heatmap: channel × time attribution
    im = axes[4].imshow(
        attr_norm,
        aspect="auto",
        origin="lower",
        extent=[time[0], time[-1], -0.5, len(channel_names) - 0.5],
    )
    axes[4].set_yticks(range(len(channel_names)))
    axes[4].set_yticklabels(channel_names)
    axes[4].set_ylabel("IG map")
    fig.colorbar(im, ax=axes[4], orientation="vertical", fraction=0.025, pad=0.01)

    # Aggregated temporal attribution curve
    axes[5].plot(time, total_time, linewidth=1.0)
    axes[5].set_ylabel("Total IG")
    axes[5].set_xlabel("Time (s)")
    axes[5].grid(True, alpha=0.25)

    fig.suptitle(f"PFF-ResNet temporal Integrated Gradients for {target_name}, sample {sample_idx}")
    fig.tight_layout()

    png = outdir / f"better_temporal_ig_{target_name.lower()}_sample{sample_idx}.png"
    pdf = outdir / f"better_temporal_ig_{target_name.lower()}_sample{sample_idx}.pdf"

    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", png)
    print("Saved:", pdf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_physfusion_seed2026.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/physfusion_seed2026/best_resnet1d_physfusion.pt")
    parser.add_argument("--model_name", default="resnet1d_physfusion")
    parser.add_argument("--max_samples", type=int, default=80)
    parser.add_argument("--num_best", type=int, default=5)
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--outdir", default="figures/ig_temporal_candidates")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    print("Using device:", device)
    print("Loading config:", args.config)
    print("Loading checkpoint:", args.checkpoint)

    cfg = load_yaml(args.config)
    ckpt = torch.load(args.checkpoint, map_location=device)

    model = build_pff_model(cfg, args.model_name)
    model = load_model_weights(model, ckpt)
    model.to(device)
    model.eval()

    ds = make_dataset(cfg, ckpt)

    max_samples = min(args.max_samples, len(ds))
    rows = []
    stored = {}

    print(f"Scanning {max_samples} samples...")

    for idx in range(max_samples):
        for target_index, target_name in [(0, "SBP"), (1, "DBP")]:
            x, physio, y, meta = get_sample(ds, idx, device)
            attr_x, attr_physio = compute_ig(model, x, physio, target_index, args.n_steps)

            stats = compute_temporal_stats(attr_x)
            stats["score"] = score_candidate(stats)
            stats["sample_index"] = idx
            stats["target"] = target_name
            rows.append(stats)

            x_np = x.detach().cpu().squeeze(0).numpy()
            stored[(idx, target_name)] = (x_np, attr_x)

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{max_samples}")

    df = pd.DataFrame(rows)
    csv_path = outdir / "temporal_ig_candidate_scores.csv"
    df.to_csv(csv_path, index=False)
    print("Saved:", csv_path)

    print("\nBest candidates by target:")
    for target_name in ["SBP", "DBP"]:
        sub = df[df["target"] == target_name].sort_values("score").head(args.num_best)
        print("\n", target_name)
        print(sub[["sample_index", "early_0_5s_fraction", "peak_second", "peak_1s_fraction", "temporal_entropy", "score"]])

        for _, row in sub.iterrows():
            idx = int(row["sample_index"])
            x_np, attr_x = stored[(idx, target_name)]
            plot_clean_temporal_ig(x_np, attr_x, target_name, idx, outdir)

    print("\nDone. Check folder:")
    print(outdir)


if __name__ == "__main__":
    main()
