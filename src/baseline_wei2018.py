from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, spectrogram

from baseline import reconstruct_received_iq
from config import ParameterScaler
from dataset import load_manifest


@dataclass(frozen=True)
class Wei2018Prediction:
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


def estimate_period_from_sequence(sequence: np.ndarray, min_lag: int, max_lag: int) -> int:
    values = np.asarray(sequence, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return int(min_lag)

    centered = values - float(np.mean(values))
    if np.allclose(centered, 0.0):
        return int(min_lag)

    autocorr = np.correlate(centered, centered, mode="full")[values.size - 1 :]
    low = max(int(min_lag), 1)
    high = min(int(max_lag), values.size - 1)
    if high <= low:
        return low

    search = autocorr[low : high + 1]
    peaks, _ = find_peaks(search)
    if peaks.size > 0:
        local_index = int(peaks[np.argmax(search[peaks])])
        return low + local_index
    return low + int(np.argmax(search))


def estimate_parameters_wei2018(
    received_iq: np.ndarray,
    sample_rate_hz: float,
    scaler: ParameterScaler,
    *,
    min_timing_gap_us: float = 0.2,
) -> Wei2018Prediction:
    energy_sequence = _compute_stft_time_marginal(received_iq, sample_rate_hz)

    min_period_samples = max(2, int(round(scaler.sampling_interval_us.min_value * 1e-6 * sample_rate_hz)))
    max_period_samples = min(
        max(len(energy_sequence) - 1, min_period_samples),
        int(round(scaler.sampling_interval_us.max_value * 1e-6 * sample_rate_hz)),
    )
    period_stft = estimate_period_from_sequence(energy_sequence, min_period_samples, max_period_samples)

    sequence_mask = _threshold_sequence(energy_sequence)
    if sequence_mask.size == 0:
        return Wei2018Prediction(
            slice_width_us=scaler.slice_width_us.min_value,
            sampling_interval_us=max(
                scaler.sampling_interval_us.min_value,
                scaler.slice_width_us.min_value + min_timing_gap_us,
            ),
            modulation_floor=scaler.modulation_floor.min_value,
        )

    period_stft = min(period_stft, sequence_mask.size)
    phase_profile = _fold_profile(sequence_mask.astype(np.float64), period_stft)
    high_mask = phase_profile >= 0.5
    run_length_stft = _longest_circular_run(high_mask)
    if run_length_stft <= 0:
        run_length_stft = 1

    stft_hop_samples = _resolve_stft_hop_samples(len(received_iq), len(energy_sequence))
    sampling_interval_us = period_stft * stft_hop_samples / sample_rate_hz * 1e6
    slice_width_us = run_length_stft * stft_hop_samples / sample_rate_hz * 1e6

    slice_width_us = float(np.clip(slice_width_us, scaler.slice_width_us.min_value, scaler.slice_width_us.max_value))
    sampling_interval_us = float(
        np.clip(
            sampling_interval_us,
            max(slice_width_us + min_timing_gap_us, scaler.sampling_interval_us.min_value),
            scaler.sampling_interval_us.max_value,
        )
    )

    return Wei2018Prediction(
        slice_width_us=slice_width_us,
        sampling_interval_us=sampling_interval_us,
        modulation_floor=scaler.modulation_floor.min_value,
    )


def evaluate_dataset_wei2018(
    data_path: str | Path,
    *,
    min_timing_gap_us: float = 0.2,
) -> dict:
    data_path = Path(data_path)
    bundle = np.load(data_path)
    labels = bundle["labels"].astype(np.float32)
    iq_features = bundle["iq"].astype(np.float32)

    manifest = load_manifest(data_path.parent)
    input_scale = float(manifest.get("input_scale", 1.0))
    dataset_config = manifest["dataset_config"]
    scaler = _build_scaler(dataset_config["scaler"])
    sample_rate_hz = float(dataset_config["sample_rate_hz"])

    predictions = np.zeros_like(labels, dtype=np.float32)
    for index in range(labels.shape[0]):
        received_iq = reconstruct_received_iq(iq_features[index], input_scale)
        prediction = estimate_parameters_wei2018(
            received_iq,
            sample_rate_hz=sample_rate_hz,
            scaler=scaler,
            min_timing_gap_us=min_timing_gap_us,
        )
        predictions[index] = prediction.as_array()

    abs_error = np.abs(predictions - labels)
    hit_tl = abs_error[:, 0] <= 0.15
    hit_ts = abs_error[:, 1] <= 0.25
    hit_x = abs_error[:, 2] <= 0.05
    joint_hit = hit_tl & hit_ts & hit_x

    counts_per_split = manifest.get("jnr_counts_per_split", {})
    jnr_values = manifest.get("jnr_values_db", [])
    repeats = int(counts_per_split.get(data_path.stem, 0))
    ordered_jnr = np.repeat(np.array(jnr_values, dtype=np.float32), repeats)[: labels.shape[0]]

    per_jnr = []
    for jnr_db in jnr_values:
        mask = ordered_jnr == np.float32(jnr_db)
        if not np.any(mask):
            continue
        per_jnr.append(
            {
                "jnr_db": float(jnr_db),
                "slice_width_hit_rate": float(np.mean(hit_tl[mask])),
                "sampling_interval_hit_rate": float(np.mean(hit_ts[mask])),
                "modulation_floor_hit_rate": float(np.mean(hit_x[mask])),
                "joint_hit_rate": float(np.mean(joint_hit[mask])),
            }
        )

    return {
        "metrics": {
            "slice_width_mae_us": float(np.mean(abs_error[:, 0])),
            "sampling_interval_mae_us": float(np.mean(abs_error[:, 1])),
            "modulation_floor_mae": float(np.mean(abs_error[:, 2])),
        },
        "overall_test_hit_rates": {
            "slice_width_hit_rate": float(np.mean(hit_tl)),
            "sampling_interval_hit_rate": float(np.mean(hit_ts)),
            "modulation_floor_hit_rate": float(np.mean(hit_x)),
            "joint_hit_rate": float(np.mean(joint_hit)),
        },
        "per_jnr_test_hit_rates": per_jnr,
    }


def save_report(report: dict, output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def _compute_stft_time_marginal(received_iq: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    sequence = np.asarray(received_iq, dtype=np.complex64)
    nperseg = min(256, max(64, int(round(len(sequence) / 32))))
    noverlap = int(round(nperseg * 0.875))
    f, t, spec = spectrogram(
        sequence,
        fs=sample_rate_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=max(512, nperseg),
        mode="magnitude",
        return_onesided=False,
    )
    return np.sum(np.abs(spec), axis=0).astype(np.float64)


def _threshold_sequence(sequence: np.ndarray) -> np.ndarray:
    values = np.asarray(sequence, dtype=np.float64)
    if values.size == 0:
        return np.zeros(0, dtype=bool)
    low = float(np.quantile(values, 0.2))
    high = float(np.quantile(values, 0.8))
    if high <= low:
        return values >= float(np.mean(values))
    threshold = 0.5 * (low + high)
    return values >= threshold


def _fold_profile(values: np.ndarray, period_samples: int) -> np.ndarray:
    period_samples = max(int(period_samples), 1)
    phase = np.arange(values.size, dtype=np.int32) % period_samples
    sums = np.bincount(phase, weights=values, minlength=period_samples).astype(np.float64)
    counts = np.bincount(phase, minlength=period_samples).astype(np.float64)
    return sums / np.maximum(counts, 1.0)


def _longest_circular_run(mask: np.ndarray) -> int:
    binary = np.asarray(mask, dtype=bool)
    if binary.size == 0 or not np.any(binary):
        return 0
    doubled = np.concatenate([binary, binary])
    best = 0
    current = 0
    for value in doubled:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return min(best, binary.size)


def _resolve_stft_hop_samples(signal_length: int, marginal_length: int) -> int:
    if marginal_length <= 1:
        return 1
    return max(int(round(signal_length / marginal_length)), 1)


def _build_scaler(raw_scaler: dict) -> ParameterScaler:
    from baseline import _build_range

    return ParameterScaler(
        slice_width_us=_build_range(raw_scaler["slice_width_us"]),
        sampling_interval_us=_build_range(raw_scaler["sampling_interval_us"]),
        modulation_floor=_build_range(raw_scaler["modulation_floor"]),
    )
