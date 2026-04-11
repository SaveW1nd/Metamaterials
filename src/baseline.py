from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import ParameterScaler
from dataset import load_manifest


@dataclass(frozen=True)
class BaselinePrediction:
    slice_width_us: float
    sampling_interval_us: float
    modulation_floor: float

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.slice_width_us,
                self.sampling_interval_us,
                self.modulation_floor,
            ],
            dtype=np.float32,
        )


def reconstruct_received_iq(iq_features: np.ndarray, input_scale: float) -> np.ndarray:
    scale = max(float(input_scale), 1e-6)
    return (iq_features[0] + 1j * iq_features[1]).astype(np.complex64) * np.complex64(scale)


def estimate_isrj_parameters(
    received_iq: np.ndarray,
    sample_rate_hz: float,
    scaler: ParameterScaler,
    min_timing_gap_us: float = 0.2,
) -> BaselinePrediction:
    envelope = np.abs(received_iq).astype(np.float64)
    smoothed = _moving_average(envelope, window=9)
    edge = _moving_average(np.abs(np.diff(smoothed, prepend=smoothed[0])), window=7)

    min_period_samples = max(1, int(round(scaler.sampling_interval_us.min_value * 1e-6 * sample_rate_hz)))
    max_period_samples = min(
        len(smoothed) - 1,
        int(round(scaler.sampling_interval_us.max_value * 1e-6 * sample_rate_hz)),
    )
    period_samples = _estimate_period_samples(edge, min_period_samples, max_period_samples)
    profile = _fold_profile(smoothed, period_samples)
    run_mask = _detect_high_region(profile)

    width_samples = max(int(np.sum(run_mask)), 1)
    slice_width_us = width_samples / sample_rate_hz * 1e6
    sampling_interval_us = period_samples / sample_rate_hz * 1e6

    max_slice_width_us = min(
        scaler.slice_width_us.max_value,
        sampling_interval_us - min_timing_gap_us,
    )
    slice_width_us = float(np.clip(slice_width_us, scaler.slice_width_us.min_value, max_slice_width_us))
    sampling_interval_us = float(
        np.clip(
            sampling_interval_us,
            max(slice_width_us + min_timing_gap_us, scaler.sampling_interval_us.min_value),
            scaler.sampling_interval_us.max_value,
        )
    )

    high_level = float(np.median(profile[run_mask])) if np.any(run_mask) else float(np.max(profile))
    low_level = float(np.median(profile[~run_mask])) if np.any(~run_mask) else float(np.min(profile))
    modulation_floor = _estimate_modulation_floor(low_level, high_level)
    modulation_floor = float(
        np.clip(
            modulation_floor,
            scaler.modulation_floor.min_value,
            scaler.modulation_floor.max_value,
        )
    )

    return BaselinePrediction(
        slice_width_us=slice_width_us,
        sampling_interval_us=sampling_interval_us,
        modulation_floor=modulation_floor,
    )


def evaluate_dataset(
    data_path: str | Path,
    manifest: dict | None = None,
    min_timing_gap_us: float = 0.2,
) -> dict:
    data_path = Path(data_path)
    bundle = np.load(data_path)
    labels = bundle["labels"].astype(np.float32)
    iq = bundle["iq"].astype(np.float32)

    if manifest is None:
        manifest = load_manifest(data_path.parent)

    input_scale = float(manifest.get("input_scale", 1.0))
    dataset_config = manifest["dataset_config"]
    scaler = _build_scaler(dataset_config["scaler"])
    sample_rate_hz = float(dataset_config["sample_rate_hz"])

    predictions = np.zeros_like(labels, dtype=np.float32)
    start = time.perf_counter()
    for index in range(len(labels)):
        received_iq = reconstruct_received_iq(iq[index], input_scale)
        prediction = estimate_isrj_parameters(
            received_iq,
            sample_rate_hz=sample_rate_hz,
            scaler=scaler,
            min_timing_gap_us=min_timing_gap_us,
        )
        predictions[index] = prediction.as_array()
    elapsed = time.perf_counter() - start

    abs_error = np.abs(predictions - labels)
    sq_error = (predictions - labels) ** 2
    hit_tl = abs_error[:, 0] <= 0.15
    hit_ts = abs_error[:, 1] <= 0.25
    hit_x = abs_error[:, 2] <= 0.05

    return {
        "num_samples": int(labels.shape[0]),
        "slice_width": _summarize_error(abs_error[:, 0], sq_error[:, 0], hit_tl),
        "sampling_interval": _summarize_error(abs_error[:, 1], sq_error[:, 1], hit_ts),
        "modulation_floor": _summarize_error(abs_error[:, 2], sq_error[:, 2], hit_x),
        "joint_hit_rate": float(np.mean(hit_tl & hit_ts & hit_x)),
        "runtime_seconds_total": float(elapsed),
        "runtime_ms_per_sample": float(elapsed * 1000.0 / max(labels.shape[0], 1)),
    }


def evaluate_all_splits(
    data_dir: str | Path,
    min_timing_gap_us: float = 0.2,
) -> dict:
    data_dir = Path(data_dir)
    manifest = load_manifest(data_dir)
    results = {}
    for split in ("train", "val", "test"):
        split_path = data_dir / f"{split}.npz"
        if split_path.exists():
            results[split] = evaluate_dataset(
                split_path,
                manifest=manifest,
                min_timing_gap_us=min_timing_gap_us,
            )
    return results


def save_evaluation_report(report: dict, output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")


def _build_scaler(raw_scaler: dict) -> ParameterScaler:
    return ParameterScaler(
        slice_width_us=_build_range(raw_scaler["slice_width_us"]),
        sampling_interval_us=_build_range(raw_scaler["sampling_interval_us"]),
        modulation_floor=_build_range(raw_scaler["modulation_floor"]),
    )


def _build_range(raw_range: dict):
    from config import RangeConfig

    return RangeConfig(
        min_value=float(raw_range["min_value"]),
        max_value=float(raw_range["max_value"]),
    )


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="same")


def _estimate_period_samples(edge_signal: np.ndarray, min_period_samples: int, max_period_samples: int) -> int:
    centered = edge_signal - np.mean(edge_signal)
    if np.allclose(centered, 0.0):
        return min_period_samples

    n = len(centered)
    fft_size = 1 << int(np.ceil(np.log2(max(2 * n, 1))))
    spectrum = np.fft.rfft(centered, n=fft_size)
    acf = np.fft.irfft(spectrum * np.conj(spectrum), n=fft_size)[:n]
    acf = acf / max(float(acf[0]), 1e-6)
    search = acf[min_period_samples : max_period_samples + 1]
    best_index = int(np.argmax(search))
    return int(min_period_samples + best_index)


def _fold_profile(values: np.ndarray, period_samples: int) -> np.ndarray:
    period_samples = max(int(period_samples), 1)
    phase = np.arange(len(values), dtype=np.int32) % period_samples
    sums = np.bincount(phase, weights=values, minlength=period_samples).astype(np.float64)
    counts = np.bincount(phase, minlength=period_samples).astype(np.float64)
    return sums / np.maximum(counts, 1.0)


def _detect_high_region(profile: np.ndarray) -> np.ndarray:
    low = float(np.quantile(profile, 0.1))
    high = float(np.quantile(profile, 0.9))
    if high <= low:
        return np.zeros_like(profile, dtype=bool)

    threshold = 0.5 * (low + high)
    binary = profile >= threshold
    if np.all(binary):
        binary[np.argmin(profile)] = False
    if not np.any(binary):
        binary[np.argmax(profile)] = True

    best_start = 0
    best_len = 0
    current_start = None
    current_len = 0
    doubled = np.concatenate([binary, binary])
    for idx, value in enumerate(doubled):
        if value:
            if current_start is None:
                current_start = idx
                current_len = 1
            else:
                current_len += 1
            if current_len > best_len and current_len <= len(binary):
                best_len = current_len
                best_start = current_start
        else:
            current_start = None
            current_len = 0

    mask = np.zeros_like(binary, dtype=bool)
    for offset in range(best_len):
        mask[(best_start + offset) % len(binary)] = True
    return mask


def _estimate_modulation_floor(low_level: float, high_level: float) -> float:
    high_delta = max(high_level - 1.0, 1e-6)
    return (low_level - 1.0) / high_delta


def _summarize_error(abs_error: np.ndarray, sq_error: np.ndarray, hits: np.ndarray) -> dict:
    return {
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean(sq_error))),
        "median_abs_error": float(np.median(abs_error)),
        "p95_abs_error": float(np.quantile(abs_error, 0.95)),
        "hit_rate": float(np.mean(hits)),
    }
