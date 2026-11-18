import csv
import time
import tempfile
import platform
from pathlib import Path

import torch


def get_gpu_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_size_mb(model):
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save(model.state_dict(), f.name)
        size_mb = Path(f.name).stat().st_size / (1024 ** 2)
    return size_mb


def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def get_peak_memory_gb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0.0


def get_batch_size_from_batch(batch):
    """
    Tries to automatically detect batch size from common batch formats.
    Works for tuple/list/dict batches where first tensor has shape [B, ...].
    """
    if isinstance(batch, dict):
        for value in batch.values():
            if torch.is_tensor(value):
                return value.shape[0]
    elif isinstance(batch, (tuple, list)):
        for value in batch:
            if torch.is_tensor(value):
                return value.shape[0]

    raise ValueError("Could not detect batch size from batch. Please set it manually.")


def benchmark_inference(
    model,
    dataloader,
    device,
    forward_batch_fn,
    warmup_batches=5,
    timed_batches=50,
):
    """
    Measures model inference speed.

    forward_batch_fn must be a function like:
        outputs = forward_batch_fn(model, batch, device)

    This keeps the profiler independent of your model type.
    """
    model.eval()

    cached_batches = []
    for i, batch in enumerate(dataloader):
        if i >= warmup_batches + timed_batches:
            break
        cached_batches.append(batch)

    if len(cached_batches) == 0:
        raise ValueError("Dataloader is empty.")

    # Warmup
    with torch.no_grad():
        for batch in cached_batches[:warmup_batches]:
            _ = forward_batch_fn(model, batch, device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_segments = 0
    start_time = time.perf_counter()

    with torch.no_grad():
        for batch in cached_batches[warmup_batches:]:
            batch_size = get_batch_size_from_batch(batch)
            total_segments += batch_size
            _ = forward_batch_fn(model, batch, device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed_seconds = time.perf_counter() - start_time

    latency_ms_per_segment = (elapsed_seconds / total_segments) * 1000
    throughput_segments_per_second = total_segments / elapsed_seconds

    return {
        "inference_time_ms_per_segment": latency_ms_per_segment,
        "throughput_segments_per_second": throughput_segments_per_second,
        "timed_inference_segments": total_segments,
        "timed_inference_seconds": elapsed_seconds,
    }


def save_computation_report(row, output_csv):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    file_exists = output_csv.exists()

    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def build_computation_row(
    model_name,
    seed,
    model,
    batch_size,
    epochs_trained,
    best_epoch,
    total_training_seconds,
    avg_epoch_seconds,
    inference_report,
):
    total_params, trainable_params = count_parameters(model)

    gpu_name = get_gpu_name()
    peak_memory_gb = get_peak_memory_gb()
    model_size_mb = get_model_size_mb(model)

    training_hours = total_training_seconds / 3600.0

    row = {
        "model_name": model_name,
        "seed": seed,
        "gpu_name": gpu_name,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,


        "batch_size": batch_size,
        "epochs_trained": epochs_trained,
        "best_epoch": best_epoch,

        "total_params_million": total_params / 1e6,
        "trainable_params_million": trainable_params / 1e6,
        "model_size_mb": model_size_mb,

        "total_training_seconds": total_training_seconds,
        "total_training_hours": training_hours,
        "avg_epoch_seconds": avg_epoch_seconds,
        "peak_gpu_memory_gb": peak_memory_gb,

        "inference_time_ms_per_segment": inference_report["inference_time_ms_per_segment"],
        "throughput_segments_per_second": inference_report["throughput_segments_per_second"],
        "timed_inference_segments": inference_report["timed_inference_segments"],
    }

    return row

