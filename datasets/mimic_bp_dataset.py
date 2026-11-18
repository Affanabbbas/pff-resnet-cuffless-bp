from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


SIGNAL_DIR_CANDIDATES = {
    "ecg": ["ecg", "ECG", "ekg", "EKG"],
    "ppg": ["ppg", "PPG", "pleth", "PLETH"],
    "labels": ["labels", "Labels", "label", "BP", "bp", "target", "targets"],
}

SPLIT_FILE_CANDIDATES = {
    "train": ["train_subjects.txt", "train.txt", "subjects_train.txt"],
    "val": ["val_subjects.txt", "validation_subjects.txt", "valid_subjects.txt", "val.txt", "validation.txt"],
    "test": ["test_subjects.txt", "test.txt", "subjects_test.txt"],
}

DEFAULT_ECG_BANDPASS = (0.5, 40.0)
DEFAULT_PPG_BANDPASS = (0.5, 8.0)


def _norm_subject_id(x: str) -> str:
    """Normalize subject IDs while preserving the p-prefix if present."""
    x = Path(str(x).strip()).stem
    x = x.replace("\ufeff", "")
    return x


def read_subject_split(data_dir: str | Path, split: str, split_dir: Optional[str | Path] = None) -> List[str]:
    """
    Read a subject split file.

    Supported formats:
        p000188
        p000188, p000333
        ['p000188', 'p000333']
    """
    import ast

    split = split.lower()
    if split in {"valid", "validation"}:
        split = "val"

    if split not in SPLIT_FILE_CANDIDATES:
        raise ValueError(f"Unknown split: {split}. Use train, val, or test.")

    roots = []
    if split_dir is not None:
        roots.append(Path(split_dir))

    roots.extend([
        Path(data_dir),
        Path(data_dir) / "splits",
        Path(data_dir) / "Split",
        Path(data_dir) / "split",
    ])

    checked = []
    for root in roots:
        for fname in SPLIT_FILE_CANDIDATES[split]:
            path = root / fname
            checked.append(str(path))

            if not path.exists():
                continue

            text = path.read_text(encoding="utf-8").replace("\ufeff", "").strip()
            if not text:
                raise ValueError(f"Split file found but empty: {path}")

            subjects: List[str] = []
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple, set)):
                    subjects = [_norm_subject_id(str(x)) for x in parsed]
                elif isinstance(parsed, str):
                    subjects = [_norm_subject_id(parsed)]
            except Exception:
                subjects = []

            if not subjects:
                subjects = re.findall(r"p\d+", text, flags=re.IGNORECASE)
                subjects = [_norm_subject_id(s.lower()) for s in subjects]

            if not subjects:
                cleaned = text
                for ch in "[](){}'\",;":
                    cleaned = cleaned.replace(ch, " ")
                for token in cleaned.split():
                    token = token.strip()
                    if token and not token.startswith("#"):
                        subjects.append(_norm_subject_id(token))

            subjects = list(dict.fromkeys(subjects))
            if not subjects:
                raise ValueError(f"No subject IDs could be parsed from split file: {path}")

            print(f"✅ Loaded {len(subjects)} subjects from {path}")
            return subjects

    raise FileNotFoundError(
        f"Could not find split file for split='{split}'. Checked:\n" + "\n".join(checked)
    )


def find_data_dir(data_dir: str | Path, kind: str) -> Path:
    """Locate ECG/PPG/labels folder and report zipped data clearly."""
    data_dir = Path(data_dir)
    candidates = SIGNAL_DIR_CANDIDATES[kind]
    for name in candidates:
        p = data_dir / name
        if p.exists() and p.is_dir():
            return p

    zip_candidates = []
    for name in candidates:
        zip_candidates.extend(list(data_dir.glob(f"{name}.zip")))
        zip_candidates.extend(list(data_dir.glob(f"{name}_files.zip")))
        zip_candidates.extend(list(data_dir.glob(f"{name}*.zip")))

    if zip_candidates:
        raise FileNotFoundError(
            f"Found zip file(s) for {kind}: {[str(z) for z in zip_candidates]}\n"
            f"Please unzip them first, e.g.:\n"
            f"  python scripts/unzip_mimic_bp.py --data_dir {data_dir}\n"
        )

    raise FileNotFoundError(
        f"Could not find a folder for '{kind}'. Expected one of: {candidates} inside {data_dir}"
    )


def find_subject_file(folder: Path, subject_id: str) -> Path:
    """
    Find file for a subject ID inside a signal/label folder.

    Supports names such as:
        p093833.npy
        p093833.dat
        p093833_ECG.npy
        ECG_p093833.dat
    """
    subject_id = _norm_subject_id(subject_id)
    exts = [".npy", ".npz", ".csv", ".txt", ".dat", ".bin"]

    direct = []
    for ext in exts:
        direct.append(folder / f"{subject_id}{ext}")
        direct.append(folder / f"{subject_id.upper()}{ext}")
        direct.append(folder / f"{subject_id.lower()}{ext}")

    for p in direct:
        if p.exists():
            return p

    matches = []
    for ext in exts:
        matches.extend(folder.glob(f"*{subject_id}*{ext}"))
        matches.extend(folder.glob(f"*{subject_id.upper()}*{ext}"))
        matches.extend(folder.glob(f"*{subject_id.lower()}*{ext}"))

    matches = sorted(set(matches))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        for m in matches:
            if _norm_subject_id(m.stem) == subject_id:
                return m
        return matches[0]

    raise FileNotFoundError(f"No file found for subject '{subject_id}' in folder {folder}")


def _load_npz_array(path: Path, label_mode: bool = False) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())

    if label_mode:
        lower = {k.lower(): k for k in keys}
        sbp_key = next((lower[k] for k in lower if "sbp" in k or "systolic" in k), None)
        dbp_key = next((lower[k] for k in lower if "dbp" in k or "diastolic" in k), None)
        if sbp_key and dbp_key:
            return np.stack(
                [np.asarray(data[sbp_key]).reshape(-1), np.asarray(data[dbp_key]).reshape(-1)],
                axis=1,
            )

        for preferred in ["labels", "label", "y", "target", "targets", "bp"]:
            if preferred in lower:
                return np.asarray(data[lower[preferred]])

    for preferred in ["data", "signal", "signals", "x", "arr_0"]:
        if preferred in keys:
            return np.asarray(data[preferred])
    return np.asarray(data[keys[0]])


def load_array_file(
    path: str | Path,
    expected_shape: Optional[Tuple[int, int]] = None,
    label_mode: bool = False,
) -> np.ndarray:
    """
    Load one subject's signal or label file.

    Supports:
        .npy, .npz, .csv, .txt, .dat, .bin

    For official-like MIMIC-BP binary signal files, this supports float64 raw
    files and files with a 128-byte header.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(path, allow_pickle=True)
    elif suffix == ".npz":
        arr = _load_npz_array(path, label_mode=label_mode)
    elif suffix in {".csv", ".txt"}:
        try:
            import pandas as pd

            df = pd.read_csv(path)
            if label_mode:
                cols = {c.lower(): c for c in df.columns}
                sbp = next((cols[c] for c in cols if "sbp" in c or "systolic" in c), None)
                dbp = next((cols[c] for c in cols if "dbp" in c or "diastolic" in c), None)
                if sbp and dbp:
                    arr = df[[sbp, dbp]].to_numpy()
                else:
                    arr = df.select_dtypes(include=[np.number]).to_numpy()
            else:
                arr = df.select_dtypes(include=[np.number]).to_numpy()
        except Exception:
            arr = np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
    elif suffix in {".dat", ".bin"}:
        if expected_shape is None:
            arr = np.fromfile(path, dtype=np.float64)
        else:
            n = int(np.prod(expected_shape))
            arr = np.fromfile(path, dtype=np.float64)
            if arr.size != n:
                with open(path, "rb") as f:
                    f.seek(128)
                    arr2 = np.fromfile(f, dtype=np.float64)
                if arr2.size == n:
                    arr = arr2
                else:
                    arr3 = np.fromfile(path, dtype=np.float32)
                    if arr3.size == n:
                        arr = arr3.astype(np.float64)
                    else:
                        raise ValueError(
                            f"Could not reshape binary file {path}. "
                            f"Expected {n} values for shape {expected_shape}, got "
                            f"float64={arr.size}, float64_after_128_header={arr2.size}, float32={arr3.size}."
                        )
            arr = arr.reshape(expected_shape)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    return np.asarray(arr)


def ensure_signal_shape(arr: np.ndarray, expected_segments: int, length: int, path: Path) -> np.ndarray:
    """Convert signal array to shape [segments, length]."""
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)

    if arr.ndim == 1:
        if arr.size == expected_segments * length:
            arr = arr.reshape(expected_segments, length)
        elif arr.size == length:
            arr = arr.reshape(1, length)
        else:
            raise ValueError(
                f"Signal {path} is 1D with length {arr.size}; expected {length} or {expected_segments * length}."
            )
    elif arr.ndim == 2:
        if arr.shape == (expected_segments, length):
            pass
        elif arr.shape == (length, expected_segments):
            arr = arr.T
        elif arr.shape[0] == 1 and arr.shape[1] == expected_segments * length:
            arr = arr.reshape(expected_segments, length)
        elif arr.shape[1] == 1 and arr.shape[0] == expected_segments * length:
            arr = arr.reshape(expected_segments, length)
        else:
            raise ValueError(
                f"Signal {path} has shape {arr.shape}; expected ({expected_segments}, {length}) "
                f"or ({length}, {expected_segments})."
            )
    else:
        raise ValueError(f"Signal {path} has unsupported ndim={arr.ndim}, shape={arr.shape}")

    return arr.astype(np.float32)


def ensure_label_shape(arr: np.ndarray, expected_segments: int, path: Path) -> np.ndarray:
    """Convert labels to [segments, 2] = [SBP, DBP]."""
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.dtype == object:
        arr = np.asarray(arr.tolist())

    if arr.ndim == 1:
        if arr.size == expected_segments * 2:
            arr = arr.reshape(expected_segments, 2)
        elif arr.size == 2:
            arr = arr.reshape(1, 2)
        else:
            raise ValueError(
                f"Label {path} is 1D with length {arr.size}; expected 2 or {expected_segments * 2}."
            )
    elif arr.ndim == 2:
        if arr.shape[1] >= 2:
            arr = arr[:, :2]
        elif arr.shape[0] >= 2:
            arr = arr.T[:, :2]
        else:
            raise ValueError(f"Label {path} has unsupported shape {arr.shape}")
    else:
        raise ValueError(f"Label {path} has unsupported ndim={arr.ndim}, shape={arr.shape}")

    arr = arr.astype(np.float32)
    if np.nanmax(arr) < 5:
        print(
            f"⚠️ Warning: Labels in {path} look normalized, max={np.nanmax(arr):.4f}. "
            "This baseline expects real BP units in mmHg for evaluation."
        )

    return arr


def _replace_nonfinite_per_segment(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).copy()
    for i in range(x.shape[0]):
        row = x[i]
        ok = np.isfinite(row)
        if ok.all():
            continue
        fill = float(np.nanmedian(row[ok])) if ok.any() else 0.0
        row[~ok] = fill
    return x


def _sos_bandpass(
    x: np.ndarray,
    sampling_rate: int,
    low_hz: float,
    high_hz: float,
    order: int = 3,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass along each segment."""
    from scipy.signal import butter, sosfiltfilt

    nyq = 0.5 * float(sampling_rate)
    low = max(float(low_hz), 1e-3) / nyq
    high = min(float(high_hz), nyq - 1e-3) / nyq
    if low >= high:
        return x.astype(np.float32)

    sos = butter(order, [low, high], btype="bandpass", output="sos")
    padlen = min(x.shape[1] - 1, 3 * (2 * len(sos) + 1))
    if padlen <= 0:
        return x.astype(np.float32)

    return sosfiltfilt(sos, x, axis=1, padlen=padlen).astype(np.float32)


def _winsorize_segments(x: np.ndarray, lower: float = 0.5, upper: float = 99.5) -> np.ndarray:
    lo = np.percentile(x, lower, axis=1, keepdims=True)
    hi = np.percentile(x, upper, axis=1, keepdims=True)
    return np.clip(x, lo, hi).astype(np.float32)


def preprocess_waveforms(
    ecg: np.ndarray,
    ppg: np.ndarray,
    sampling_rate: int,
    filter_signals: bool = True,
    ecg_bandpass: Tuple[float, float] = DEFAULT_ECG_BANDPASS,
    ppg_bandpass: Tuple[float, float] = DEFAULT_PPG_BANDPASS,
    winsorize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Basic waveform cleanup before model normalization.

    This intentionally stays conservative:
        - replace NaNs/infs per segment
        - optional percentile clipping for rare spikes
        - optional ECG/PPG bandpass filtering
    """
    ecg = _replace_nonfinite_per_segment(ecg)
    ppg = _replace_nonfinite_per_segment(ppg)

    if winsorize:
        ecg = _winsorize_segments(ecg)
        ppg = _winsorize_segments(ppg)

    if filter_signals:
        ecg = _sos_bandpass(ecg, sampling_rate, ecg_bandpass[0], ecg_bandpass[1])
        ppg = _sos_bandpass(ppg, sampling_rate, ppg_bandpass[0], ppg_bandpass[1])

    return ecg.astype(np.float32), ppg.astype(np.float32)


def add_ppg_derivatives(ecg: np.ndarray, ppg: np.ndarray) -> np.ndarray:
    """Build channels [ECG, PPG, dPPG, ddPPG], each [segments, length]."""
    d1 = np.gradient(ppg, axis=1).astype(np.float32)
    d2 = np.gradient(d1, axis=1).astype(np.float32)
    return np.stack([ecg, ppg, d1, d2], axis=1).astype(np.float32)


def _safe_z(x: np.ndarray) -> np.ndarray:
    mean = np.mean(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True)
    return (x - mean) / (std + 1e-8)


def _count_plausible_peaks(signal: np.ndarray, sampling_rate: int) -> np.ndarray:
    from scipy.signal import find_peaks

    counts = np.zeros(signal.shape[0], dtype=np.float32)
    min_distance = max(1, int(0.30 * sampling_rate))

    for i, seg in enumerate(signal):
        z = (seg - np.mean(seg)) / (np.std(seg) + 1e-8)
        peaks_pos, _ = find_peaks(z, distance=min_distance, prominence=0.25)
        peaks_neg, _ = find_peaks(-z, distance=min_distance, prominence=0.25)
        counts[i] = float(max(len(peaks_pos), len(peaks_neg)))

    return counts


def signal_quality_mask(
    ecg: np.ndarray,
    ppg: np.ndarray,
    y: np.ndarray,
    sampling_rate: int,
    min_ecg_std: float = 1e-4,
    min_ppg_std: float = 1e-4,
    min_peak_count: int = 10,
    max_abs_z: float = 12.0,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Return a conservative valid-segment mask.

    The thresholds avoid the most obvious failure modes: NaN/inf, impossible BP,
    flatlines, extreme spikes, and PPG segments with implausibly few pulses.
    """
    finite_x = np.isfinite(ecg).all(axis=1) & np.isfinite(ppg).all(axis=1)
    finite_y = np.isfinite(y).all(axis=1)
    y_range_ok = (y[:, 0] > 40) & (y[:, 0] < 260) & (y[:, 1] > 20) & (y[:, 1] < 180) & (y[:, 0] > y[:, 1])

    ecg_std = np.std(ecg, axis=1)
    ppg_std = np.std(ppg, axis=1)
    ecg_ptp = np.ptp(ecg, axis=1)
    ppg_ptp = np.ptp(ppg, axis=1)
    flat_ok = (ecg_std > min_ecg_std) & (ppg_std > min_ppg_std) & (ecg_ptp > min_ecg_std) & (ppg_ptp > min_ppg_std)

    ecg_max_z = np.max(np.abs(_safe_z(ecg)), axis=1)
    ppg_max_z = np.max(np.abs(_safe_z(ppg)), axis=1)
    spike_ok = (ecg_max_z < max_abs_z) & (ppg_max_z < max_abs_z)

    ppg_peak_count = _count_plausible_peaks(ppg, sampling_rate)
    peak_ok = ppg_peak_count >= float(min_peak_count)

    valid = finite_x & finite_y & y_range_ok & flat_ok & spike_ok & peak_ok
    quality = {
        "finite_x": finite_x.astype(np.float32),
        "finite_y": finite_y.astype(np.float32),
        "label_range_ok": y_range_ok.astype(np.float32),
        "ecg_std": ecg_std.astype(np.float32),
        "ppg_std": ppg_std.astype(np.float32),
        "ecg_max_z": ecg_max_z.astype(np.float32),
        "ppg_max_z": ppg_max_z.astype(np.float32),
        "ppg_peak_count": ppg_peak_count.astype(np.float32),
        "valid": valid.astype(np.float32),
    }
    return valid, quality


def basic_filter_nan(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove invalid labels/samples. Returns valid indices too."""
    x_ok = np.isfinite(x).all(axis=(1, 2))
    y_ok = np.isfinite(y).all(axis=1)
    y_range_ok = (y[:, 0] > 40) & (y[:, 0] < 260) & (y[:, 1] > 20) & (y[:, 1] < 180) & (y[:, 0] > y[:, 1])
    valid = x_ok & y_ok & y_range_ok
    return x[valid], y[valid], np.where(valid)[0]


def _find_peaks_1d(signal: np.ndarray, sampling_rate: int, prominence: float = 0.30) -> np.ndarray:
    from scipy.signal import find_peaks

    z = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    min_distance = max(1, int(0.30 * sampling_rate))
    pos, _ = find_peaks(z, distance=min_distance, prominence=prominence)
    neg, _ = find_peaks(-z, distance=min_distance, prominence=prominence)
    return pos if len(pos) >= len(neg) else neg


def extract_physiological_features(ecg: np.ndarray, ppg: np.ndarray, sampling_rate: int) -> np.ndarray:
    """
    Compute lightweight physiology-inspired features per segment.

    These are not used by the current ResNet. They are exposed in meta so a
    later fusion model can consume them without another dataset rewrite.
    """
    dppg = np.gradient(ppg, axis=1)
    ddppg = np.gradient(dppg, axis=1)
    duration_min = ecg.shape[1] / float(sampling_rate) / 60.0

    feats = []
    for i in range(ecg.shape[0]):
        ecg_i = ecg[i]
        ppg_i = ppg[i]
        dppg_i = dppg[i]
        ddppg_i = ddppg[i]

        ecg_peaks = _find_peaks_1d(ecg_i, sampling_rate, prominence=0.35)
        ppg_peaks = _find_peaks_1d(ppg_i, sampling_rate, prominence=0.25)

        hr_ecg = len(ecg_peaks) / max(duration_min, 1e-8)
        hr_ppg = len(ppg_peaks) / max(duration_min, 1e-8)

        ptt_values = []
        if len(ecg_peaks) > 0 and len(ppg_peaks) > 0:
            for r in ecg_peaks:
                after = ppg_peaks[(ppg_peaks > r + int(0.08 * sampling_rate)) & (ppg_peaks < r + int(0.60 * sampling_rate))]
                if len(after) > 0:
                    ptt_values.append((after[0] - r) / float(sampling_rate))
        median_ptt = float(np.median(ptt_values)) if ptt_values else 0.0

        feats.append([
            float(np.mean(ecg_i)),
            float(np.std(ecg_i)),
            float(np.ptp(ecg_i)),
            float(np.mean(ppg_i)),
            float(np.std(ppg_i)),
            float(np.ptp(ppg_i)),
            float(np.mean(np.abs(dppg_i))),
            float(np.mean(np.abs(ddppg_i))),
            float(hr_ecg),
            float(hr_ppg),
            float(median_ptt),
            float(len(ppg_peaks)),
        ])

    return np.asarray(feats, dtype=np.float32)


class MIMICBPDataset(Dataset):
    """
    PyTorch Dataset for MIMIC-BP.

    Returns:
        x: tensor [channels, length]
        y: tensor [2] = [SBP, DBP], optionally standardized
        meta: dict with subject_id, segment_id, and optional physio_features
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        split_dir: Optional[str | Path] = None,
        use_derivatives: bool = False,
        normalization: str = "global_train",
        stats: Optional[Dict[str, np.ndarray]] = None,
        sampling_rate: int = 125,
        segment_seconds: int = 30,
        expected_segments_per_subject: int = 30,
        max_cache_subjects: int = 16,
        verbose: bool = True,
        preprocess: bool = False,
        filter_signals: bool = True,
        quality_filter: bool = False,
        min_ppg_peak_count: int = 10,
        include_physio_features: bool = False,
        standardize_labels: bool = False,
        label_stats: Optional[Dict[str, np.ndarray]] = None,
        build_index_from_signals: Optional[bool] = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = "val" if split in {"valid", "validation"} else split
        self.use_derivatives = bool(use_derivatives)
        self.normalization = normalization
        self.stats = stats
        self.sampling_rate = int(sampling_rate)
        self.segment_seconds = int(segment_seconds)
        self.length = int(self.sampling_rate * self.segment_seconds)
        self.expected_segments = int(expected_segments_per_subject)
        self.max_cache_subjects = int(max_cache_subjects)
        self.verbose = verbose
        self.preprocess = bool(preprocess)
        self.filter_signals = bool(filter_signals)
        self.quality_filter = bool(quality_filter)
        self.min_ppg_peak_count = int(min_ppg_peak_count)
        self.include_physio_features = bool(include_physio_features)
        self.standardize_labels = bool(standardize_labels)
        self.label_stats = label_stats
        self.build_index_from_signals = self.quality_filter if build_index_from_signals is None else bool(build_index_from_signals)

        if self.normalization not in {"none", "global_train", "per_segment_zscore"}:
            raise ValueError("normalization must be one of: none, global_train, per_segment_zscore")
        if self.standardize_labels and self.label_stats is None:
            raise ValueError("standardize_labels=True requires label_stats={'mean': ..., 'std': ...}.")

        self.subjects = read_subject_split(self.data_dir, self.split, split_dir)
        self.ecg_dir = find_data_dir(self.data_dir, "ecg")
        self.ppg_dir = find_data_dir(self.data_dir, "ppg")
        self.label_dir = find_data_dir(self.data_dir, "labels")

        self.subject_files: Dict[str, Tuple[Path, Path, Path]] = {}
        self.samples: List[Tuple[str, int]] = []
        self._cache: OrderedDict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]] = OrderedDict()

        missing = []
        for sid in self.subjects:
            try:
                ecg_path = find_subject_file(self.ecg_dir, sid)
                ppg_path = find_subject_file(self.ppg_dir, sid)
                label_path = find_subject_file(self.label_dir, sid)
                self.subject_files[sid] = (ecg_path, ppg_path, label_path)

                if self.build_index_from_signals:
                    _, _, valid_idx, _ = self._load_subject(sid)
                    for seg_idx in valid_idx:
                        self.samples.append((sid, int(seg_idx)))
                else:
                    labels = self._load_labels(label_path)
                    for seg_idx in range(labels.shape[0]):
                        self.samples.append((sid, seg_idx))
            except Exception as e:
                missing.append((sid, str(e)))

        if missing:
            msg = "\n".join([f"{sid}: {err}" for sid, err in missing[:10]])
            raise FileNotFoundError(f"Missing/unreadable files for {len(missing)} subjects. Examples:\n{msg}")
        if not self.samples:
            raise ValueError(f"No usable samples found for split={self.split}.")

        if verbose:
            print(f"✅ MIMIC-BP {self.split}: {len(self.subjects)} subjects, {len(self.samples)} samples")
            print(f"   ECG dir: {self.ecg_dir}")
            print(f"   PPG dir: {self.ppg_dir}")
            print(f"   Labels dir: {self.label_dir}")
            print(f"   input channels: {self.num_channels}")
            print(f"   preprocess: {self.preprocess}, quality_filter: {self.quality_filter}")

    @property
    def num_channels(self) -> int:
        return 4 if self.use_derivatives else 2

    def _load_signal(self, path: Path) -> np.ndarray:
        arr = load_array_file(path, expected_shape=(self.expected_segments, self.length), label_mode=False)
        return ensure_signal_shape(arr, self.expected_segments, self.length, path)

    def _load_labels(self, path: Path) -> np.ndarray:
        arr = load_array_file(path, expected_shape=None, label_mode=True)
        return ensure_label_shape(arr, self.expected_segments, path)

    def _load_subject(self, subject_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if subject_id in self._cache:
            self._cache.move_to_end(subject_id)
            return self._cache[subject_id]

        ecg_path, ppg_path, label_path = self.subject_files[subject_id]
        ecg = self._load_signal(ecg_path)
        ppg = self._load_signal(ppg_path)
        y = self._load_labels(label_path)

        n = min(ecg.shape[0], ppg.shape[0], y.shape[0])
        ecg, ppg, y = ecg[:n], ppg[:n], y[:n]

        if self.preprocess:
            ecg, ppg = preprocess_waveforms(
                ecg,
                ppg,
                sampling_rate=self.sampling_rate,
                filter_signals=self.filter_signals,
            )

        if self.use_derivatives:
            x = add_ppg_derivatives(ecg, ppg)
        else:
            x = np.stack([ecg, ppg], axis=1).astype(np.float32)

        if self.quality_filter:
            valid, _ = signal_quality_mask(
                ecg,
                ppg,
                y,
                sampling_rate=self.sampling_rate,
                min_peak_count=self.min_ppg_peak_count,
            )
            valid_idx = np.where(valid)[0]
            if len(valid_idx) == 0:
                x, y, valid_idx = basic_filter_nan(x, y)
            else:
                x, y = x[valid], y[valid]
        else:
            x, y, valid_idx = basic_filter_nan(x, y)

        if len(y) == 0:
            raise ValueError(f"Subject {subject_id} has no valid BP segments after filtering.")

        features = None
        if self.include_physio_features:
            ecg_valid = x[:, 0, :]
            ppg_valid = x[:, 1, :]
            features = extract_physiological_features(ecg_valid, ppg_valid, self.sampling_rate)

        if len(self._cache) >= self.max_cache_subjects:
            self._cache.popitem(last=False)
        self._cache[subject_id] = (x, y, valid_idx.astype(np.int64), features)
        return self._cache[subject_id]

    def iter_subject_arrays(self) -> Iterable[Tuple[str, np.ndarray, np.ndarray]]:
        for sid in self.subjects:
            x, y, _, _ = self._load_subject(sid)
            yield sid, x, y

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sid, original_seg_idx = self.samples[index]
        x_all, y_all, valid_idx, features_all = self._load_subject(sid)

        positions = np.where(valid_idx == original_seg_idx)[0]
        if len(positions) == 0:
            pos = original_seg_idx % len(y_all)
            segment_id = int(valid_idx[pos])
        else:
            pos = int(positions[0])
            segment_id = int(original_seg_idx)

        x = x_all[pos].astype(np.float32)
        y = y_all[pos].astype(np.float32)

        if self.normalization == "global_train":
            if self.stats is None:
                raise ValueError("normalization='global_train' requires train statistics.")
            mean = np.asarray(self.stats["mean"], dtype=np.float32).reshape(-1, 1)
            std = np.asarray(self.stats["std"], dtype=np.float32).reshape(-1, 1)
            x = (x - mean) / (std + 1e-8)
        elif self.normalization == "per_segment_zscore":
            mean = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True)
            x = (x - mean) / (std + 1e-8)

        if self.standardize_labels:
            mean = np.asarray(self.label_stats["mean"], dtype=np.float32)
            std = np.asarray(self.label_stats["std"], dtype=np.float32)
            y = (y - mean) / (std + 1e-8)

        meta = {"subject_id": sid, "segment_id": segment_id}
        if self.include_physio_features and features_all is not None:
            meta["physio_features"] = features_all[pos].astype(np.float32)

        return torch.from_numpy(x), torch.from_numpy(y), meta


def compute_train_signal_stats(dataset: MIMICBPDataset, max_subjects: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Compute input normalization stats from TRAIN subjects only.

    Returns:
        {"mean": [channels], "std": [channels]}
    """
    if dataset.split != "train":
        raise ValueError("compute_train_signal_stats should be called on the train split only.")

    channel_sum = None
    channel_sq_sum = None
    count = 0
    n_subjects = 0

    for sid, x, _ in dataset.iter_subject_arrays():
        x = x.astype(np.float64)
        if channel_sum is None:
            channel_sum = np.zeros(x.shape[1], dtype=np.float64)
            channel_sq_sum = np.zeros(x.shape[1], dtype=np.float64)

        channel_sum += x.sum(axis=(0, 2))
        channel_sq_sum += (x ** 2).sum(axis=(0, 2))
        count += x.shape[0] * x.shape[2]
        n_subjects += 1

        if max_subjects is not None and n_subjects >= max_subjects:
            break

    mean = channel_sum / max(count, 1)
    var = channel_sq_sum / max(count, 1) - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-8))

    print(f"✅ Computed train-only signal stats from {n_subjects} subjects")
    print(f"   mean={mean}")
    print(f"   std={std}")

    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def compute_train_label_stats(dataset: MIMICBPDataset, max_subjects: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Compute SBP/DBP standardization stats from TRAIN subjects only.

    Returns:
        {"mean": [2], "std": [2]}
    """
    if dataset.split != "train":
        raise ValueError("compute_train_label_stats should be called on the train split only.")

    label_sum = np.zeros(2, dtype=np.float64)
    label_sq_sum = np.zeros(2, dtype=np.float64)
    count = 0
    n_subjects = 0

    for sid, _, y in dataset.iter_subject_arrays():
        y = y.astype(np.float64)
        label_sum += y.sum(axis=0)
        label_sq_sum += (y ** 2).sum(axis=0)
        count += y.shape[0]
        n_subjects += 1

        if max_subjects is not None and n_subjects >= max_subjects:
            break

    mean = label_sum / max(count, 1)
    var = label_sq_sum / max(count, 1) - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-8))

    print(f"✅ Computed train-only label stats from {n_subjects} subjects")
    print(f"   label_mean={mean}")
    print(f"   label_std={std}")

    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
