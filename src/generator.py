from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from config import DatasetConfig, PRIMARY_PARAM_NAMES, SignalConfig


@dataclass
class GeneratedSample:
    received_iq: np.ndarray
    target_iq: np.ndarray
    jammer_iq: np.ndarray
    jammer_mask: np.ndarray
    time_axis_s: np.ndarray
    labels: dict[str, float]
    signal_config: SignalConfig


def generate_isrj_sample(config: SignalConfig) -> GeneratedSample:
    return generate_isrj_sample_with_rng(config, rng=None)


def generate_isrj_sample_with_rng(
    config: SignalConfig,
    rng: np.random.Generator | None,
) -> GeneratedSample:
    config.validate()

    num_pulse = round(config.pulse_width_s * config.sample_rate_hz)
    tp = np.arange(num_pulse, dtype=np.float64) / config.sample_rate_hz
    time_axis = np.arange(config.num_samples, dtype=np.float64) / config.sample_rate_hz

    chirp_rate = config.bandwidth_hz / config.pulse_width_s
    lfm_pulse = np.exp(
        1j * 2.0 * np.pi * (config.carrier_hz * tp + 0.5 * chirp_rate * tp**2)
    ).astype(np.complex64)

    target_iq = _place_pulse(config.num_samples, lfm_pulse, config.sample_rate_hz, config.target_delay_s)
    jammer_echo = _place_pulse(config.num_samples, lfm_pulse, config.sample_rate_hz, config.jammer_delay_s)

    relative_t = time_axis - config.jammer_delay_s
    gate = ((relative_t >= 0.0) & (np.mod(relative_t, config.sampling_interval_s) < config.slice_width_s)).astype(
        np.float32
    )
    jammer_mask = config.modulation_floor + (1.0 - config.modulation_floor) * gate
    jammer_iq = jammer_echo * jammer_mask.astype(np.float32)

    reference_power = max(float(np.mean(np.abs(target_iq) ** 2)), 1e-12)
    jammer_power = max(float(np.mean(np.abs(jammer_iq) ** 2)), 1e-12)
    jammer_scale = np.sqrt(reference_power * 10.0 ** (config.jnr_db / 10.0) / jammer_power)
    jammer_iq = jammer_iq * np.complex64(jammer_scale)

    noise_power = reference_power / (10.0 ** (config.snr_db / 10.0))
    noise_sigma = np.sqrt(noise_power / 2.0)
    random_source = rng if rng is not None else np.random.default_rng()
    noise = noise_sigma * (
        random_source.standard_normal(config.num_samples).astype(np.float32)
        + 1j * random_source.standard_normal(config.num_samples).astype(np.float32)
    )

    received_iq = (target_iq + jammer_iq + noise).astype(np.complex64)

    labels = {
        "slice_width_us": config.slice_width_s * 1e6,
        "sampling_interval_us": config.sampling_interval_s * 1e6,
        "modulation_floor": config.modulation_floor,
    }
    return GeneratedSample(
        received_iq=received_iq,
        target_iq=target_iq.astype(np.complex64),
        jammer_iq=jammer_iq.astype(np.complex64),
        jammer_mask=jammer_mask.astype(np.float32),
        time_axis_s=time_axis.astype(np.float32),
        labels=labels,
        signal_config=config,
    )


def sample_signal_config(rng: np.random.Generator, dataset_config: DatasetConfig) -> SignalConfig:
    slice_width_us, sampling_interval_us = _sample_primary_timing(rng, dataset_config)
    modulation_floor = dataset_config.scaler.modulation_floor.sample(rng)
    snr_db = _sample_snr_db(rng, dataset_config)
    jnr_db = _sample_jnr_db(rng, dataset_config)

    return SignalConfig(
        bandwidth_hz=dataset_config.bandwidth_hz,
        pulse_width_s=dataset_config.pulse_width_s,
        carrier_hz=dataset_config.carrier_hz,
        sample_rate_hz=dataset_config.sample_rate_hz,
        num_samples=dataset_config.num_samples,
        target_delay_s=dataset_config.target_delay_s,
        jammer_delay_s=dataset_config.jammer_delay_s,
        slice_width_s=slice_width_us * 1e-6,
        sampling_interval_s=sampling_interval_us * 1e-6,
        modulation_floor=modulation_floor,
        snr_db=snr_db,
        jnr_db=jnr_db,
    )


def _sample_snr_db(rng: np.random.Generator, dataset_config: DatasetConfig) -> float:
    if dataset_config.fixed_snr_db is not None:
        return float(dataset_config.fixed_snr_db)
    return float(rng.uniform(*dataset_config.snr_db_range))


def _sample_jnr_db(rng: np.random.Generator, dataset_config: DatasetConfig) -> float:
    if dataset_config.jnr_discrete_values:
        values = dataset_config.jnr_discrete_values
        return float(values[int(rng.integers(0, len(values)))])
    return float(rng.uniform(*dataset_config.jnr_db_range))


def _sample_primary_timing(rng: np.random.Generator, dataset_config: DatasetConfig) -> tuple[float, float]:
    if dataset_config.duty_sampling_strategy == "balanced_bins":
        sampled = _sample_balanced_timing(rng, dataset_config)
        if sampled is not None:
            return sampled
    return _sample_uniform_timing(rng, dataset_config)


def _sample_uniform_timing(rng: np.random.Generator, dataset_config: DatasetConfig) -> tuple[float, float]:
    slice_width_us = dataset_config.scaler.slice_width_us.sample(rng)
    min_interval = max(
        dataset_config.scaler.sampling_interval_us.min_value,
        slice_width_us + dataset_config.min_interval_gap_us,
    )
    sampling_interval_us = float(rng.uniform(min_interval, dataset_config.scaler.sampling_interval_us.max_value))
    return slice_width_us, sampling_interval_us


def _sample_balanced_timing(
    rng: np.random.Generator,
    dataset_config: DatasetConfig,
) -> tuple[float, float] | None:
    slice_range = dataset_config.scaler.slice_width_us
    interval_range = dataset_config.scaler.sampling_interval_us
    gap_min = dataset_config.min_interval_gap_us
    duty_bins = list(dataset_config.duty_bins)
    if not duty_bins:
        return None

    for _ in range(max(dataset_config.duty_sampling_attempts, 1)):
        duty_low, duty_high = duty_bins[int(rng.integers(0, len(duty_bins)))]
        duty_low = float(duty_low)
        duty_high = float(duty_high)
        if duty_low <= 0.0 or duty_high <= duty_low:
            continue

        min_interval = max(interval_range.min_value, slice_range.min_value / duty_high)
        max_interval = interval_range.max_value
        if duty_low > 0.0:
            max_interval = min(max_interval, slice_range.max_value / duty_low)
        if min_interval >= max_interval:
            continue

        sampling_interval_us = float(rng.uniform(min_interval, max_interval))
        slice_min = max(slice_range.min_value, duty_low * sampling_interval_us)
        slice_max = min(
            slice_range.max_value,
            duty_high * sampling_interval_us,
            sampling_interval_us - gap_min,
        )
        if slice_min >= slice_max:
            continue

        slice_width_us = float(rng.uniform(slice_min, slice_max))
        return slice_width_us, sampling_interval_us

    return None


def split_complex_channels(values: np.ndarray) -> np.ndarray:
    return np.stack([values.real, values.imag], axis=0).astype(np.float32)


def compute_input_scale(received_iq: np.ndarray) -> float:
    magnitude = np.abs(received_iq).astype(np.float32)
    scale = float(np.percentile(magnitude, 99))
    return max(scale, 1e-6)


def estimate_input_scale(samples: list[GeneratedSample]) -> float:
    if not samples:
        return 1.0
    scales = np.array([compute_input_scale(sample.received_iq) for sample in samples], dtype=np.float32)
    return float(np.median(scales).item())


def build_model_input(
    received_iq: np.ndarray,
    input_scale: float,
    input_representation: str = "feature3",
    third_channel_mode: str = "mag_log",
    smoothing_window: int = 9,
) -> np.ndarray:
    scale = max(float(input_scale), 1e-6)
    if str(input_representation).lower() == "iq2":
        return np.stack(
            [
                (received_iq.real / scale).astype(np.float32),
                (received_iq.imag / scale).astype(np.float32),
            ],
            axis=0,
        )

    magnitude = np.abs(received_iq).astype(np.float32) / scale
    third_channel = _build_third_channel(
        received_iq=received_iq,
        magnitude=magnitude,
        third_channel_mode=third_channel_mode,
        smoothing_window=smoothing_window,
    )
    return np.stack(
        [
            (received_iq.real / scale).astype(np.float32),
            (received_iq.imag / scale).astype(np.float32),
            third_channel,
        ],
        axis=0,
    )


def build_dataset_arrays(
    samples: list[GeneratedSample],
    input_scale: float,
    input_representation: str = "feature3",
    third_channel_mode: str = "mag_log",
    smoothing_window: int = 9,
) -> dict[str, np.ndarray]:
    iq = np.stack(
        [
            build_model_input(
                sample.received_iq,
                input_scale,
                input_representation=input_representation,
                third_channel_mode=third_channel_mode,
                smoothing_window=smoothing_window,
            )
            for sample in samples
        ],
        axis=0,
    )
    jammer_iq = np.stack([split_complex_channels(sample.jammer_iq) for sample in samples], axis=0)
    labels = np.stack([[sample.labels[name] for name in PRIMARY_PARAM_NAMES] for sample in samples], axis=0).astype(
        np.float32
    )
    masks = np.stack([sample.jammer_mask for sample in samples], axis=0).astype(np.float32)
    return {
        "iq": iq,
        "jammer_iq": jammer_iq,
        "labels": labels,
        "jammer_mask": masks,
    }


def _place_pulse(num_samples: int, pulse: np.ndarray, sample_rate_hz: float, delay_s: float) -> np.ndarray:
    result = np.zeros(num_samples, dtype=np.complex64)
    start = int(round(delay_s * sample_rate_hz))
    if start >= num_samples:
        return result
    stop = min(num_samples, start + len(pulse))
    count = max(stop - start, 0)
    if count > 0:
        result[start:stop] = pulse[:count]
    return result


def _build_third_channel(
    received_iq: np.ndarray,
    magnitude: np.ndarray,
    third_channel_mode: str,
    smoothing_window: int,
) -> np.ndarray:
    mode = str(third_channel_mode).lower()
    if mode == "mag_log":
        return np.log1p(magnitude).astype(np.float32)
    if mode == "phase_diff":
        phase_diff = np.angle(received_iq[1:] * np.conj(received_iq[:-1])).astype(np.float32)
        phase_diff = np.concatenate(
            [
                np.zeros(1, dtype=np.float32),
                phase_diff / np.float32(np.pi),
            ]
        )
        return np.clip(phase_diff, -1.0, 1.0).astype(np.float32)
    if mode == "env_contrast":
        mag_log = np.log1p(magnitude).astype(np.float32)
        baseline = _moving_average_1d(mag_log, max(int(smoothing_window), 1))
        return (mag_log - baseline).astype(np.float32)
    raise ValueError(f"Unsupported third_channel_mode: {third_channel_mode}")


def _moving_average_1d(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, min(int(window), int(values.shape[0])))
    if window <= 1:
        return values.astype(np.float32)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values.astype(np.float32), kernel, mode="same").astype(np.float32)
