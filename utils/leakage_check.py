from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Set

from datasets.mimic_bp_dataset import read_subject_split


def load_all_splits(data_dir: str | Path, split_dir: Optional[str | Path] = None) -> Dict[str, Set[str]]:
    return {
        "train": set(read_subject_split(data_dir, "train", split_dir)),
        "val": set(read_subject_split(data_dir, "val", split_dir)),
        "test": set(read_subject_split(data_dir, "test", split_dir)),
    }


def check_subject_leakage(data_dir: str | Path, split_dir: Optional[str | Path] = None, raise_on_leakage: bool = True):
    splits = load_all_splits(data_dir, split_dir)
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]

    print("🔍 Subject-wise leakage check")
    ok = True
    for a, b in pairs:
        overlap = splits[a] & splits[b]
        print(f"   {a} vs {b}: overlap = {len(overlap)}")
        if overlap:
            ok = False
            print(f"   Example overlap IDs: {sorted(list(overlap))[:20]}")

    print(
        f"   subjects: train={len(splits['train'])}, "
        f"val={len(splits['val'])}, test={len(splits['test'])}"
    )

    if raise_on_leakage and not ok:
        raise RuntimeError("Subject leakage detected. Do not train until this is fixed.")
    return ok, splits
