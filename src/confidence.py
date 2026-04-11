from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import ParameterScaler


@dataclass(frozen=True)
class ConfidenceOutputs:
    confidence_score: np.ndarray
    risk_score: np.ndarray
    duty: np.ndarray
    gap_margin_us: np.ndarray
    boundary_margin_us: np.ndarray
    high_duty_flag: np.ndarray
    high_x_flag: np.ndarray
    tight_gap_flag: np.ndarray
    boundary_flag: np.ndarray


def compute_confidence_outputs(
    predictions: np.ndarray,
    scaler: ParameterScaler,
    min_timing_gap_us: float,
) -> ConfidenceOutputs:
    slice_width = predictions[:, 0].astype(np.float32)
    sampling_interval = predictions[:, 1].astype(np.float32)
    modulation_floor = predictions[:, 2].astype(np.float32)

    duty = slice_width / np.maximum(sampling_interval, 1e-6)
    gap_margin_us = np.maximum(sampling_interval - slice_width - min_timing_gap_us, 0.0).astype(np.float32)

    tl_lower_margin = np.maximum(slice_width - scaler.slice_width_us.min_value, 0.0)
    tl_upper_margin = np.maximum(
        np.minimum(
            scaler.slice_width_us.max_value - slice_width,
            sampling_interval - slice_width - min_timing_gap_us,
        ),
        0.0,
    )
    ts_margin = np.maximum(
        np.minimum(
            sampling_interval - scaler.sampling_interval_us.min_value,
            scaler.sampling_interval_us.max_value - sampling_interval,
        ),
        0.0,
    )
    boundary_margin_us = np.minimum(np.minimum(tl_lower_margin, tl_upper_margin), ts_margin).astype(np.float32)

    duty_risk = _ramp(duty, start=0.45, end=0.85)
    x_risk = _ramp(modulation_floor, start=0.25, end=scaler.modulation_floor.max_value)
    gap_risk = 1.0 - _ramp(gap_margin_us, start=0.15, end=1.0)
    boundary_risk = 1.0 - _ramp(boundary_margin_us, start=0.05, end=0.50)

    risk_score = (
        0.35 * duty_risk
        + 0.25 * x_risk
        + 0.25 * gap_risk
        + 0.15 * boundary_risk
    ).astype(np.float32)
    confidence_score = np.clip(1.0 - risk_score, 0.0, 1.0).astype(np.float32)

    return ConfidenceOutputs(
        confidence_score=confidence_score,
        risk_score=risk_score,
        duty=duty.astype(np.float32),
        gap_margin_us=gap_margin_us,
        boundary_margin_us=boundary_margin_us,
        high_duty_flag=duty >= 0.55,
        high_x_flag=modulation_floor >= 0.375,
        tight_gap_flag=gap_margin_us <= 0.35,
        boundary_flag=boundary_margin_us <= 0.15,
    )


def summarize_confidence_alignment(
    confidence_score: np.ndarray,
    joint_hit: np.ndarray,
) -> dict[str, float | list[dict[str, float | int]]]:
    if confidence_score.size == 0:
        return {
            "confidence_alignment_score": 0.0,
            "confidence_bins": [],
        }

    quantiles = np.quantile(confidence_score, [1.0 / 3.0, 2.0 / 3.0])
    low_mask = confidence_score <= quantiles[0]
    high_mask = confidence_score > quantiles[1]
    mid_mask = (~low_mask) & (~high_mask)

    bins = [
        _summarize_bin("low", confidence_score, joint_hit, low_mask),
        _summarize_bin("mid", confidence_score, joint_hit, mid_mask),
        _summarize_bin("high", confidence_score, joint_hit, high_mask),
    ]

    low_hit = float(bins[0]["joint_hit_rate"]) if bins[0]["count"] > 0 else 0.0
    mid_hit = float(bins[1]["joint_hit_rate"]) if bins[1]["count"] > 0 else low_hit
    high_hit = float(bins[2]["joint_hit_rate"]) if bins[2]["count"] > 0 else mid_hit
    monotonic_bonus = 0.05 if low_hit <= mid_hit <= high_hit else 0.0
    confidence_alignment_score = max(high_hit - low_hit, 0.0) + monotonic_bonus

    return {
        "confidence_alignment_score": float(confidence_alignment_score),
        "confidence_bins": bins,
    }


def _summarize_bin(
    label: str,
    confidence_score: np.ndarray,
    joint_hit: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int | str]:
    count = int(np.sum(mask))
    if count == 0:
        return {
            "label": label,
            "count": 0,
            "mean_confidence": 0.0,
            "joint_hit_rate": 0.0,
        }
    return {
        "label": label,
        "count": count,
        "mean_confidence": float(np.mean(confidence_score[mask])),
        "joint_hit_rate": float(np.mean(joint_hit[mask])),
    }


def _ramp(values: np.ndarray, start: float, end: float) -> np.ndarray:
    if end <= start:
        raise ValueError("end must be greater than start.")
    scaled = (values - start) / (end - start)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)
