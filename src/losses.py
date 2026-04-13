from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from config import ParameterScaler
from model import DirectRegressionPredictions, GateReconstructionPredictions, TwoStagePredictions


@dataclass
class LossOutput:
    total: torch.Tensor
    regression: torch.Tensor
    ordering: torch.Tensor
    consistency: torch.Tensor


@dataclass
class DecodedPredictions:
    ts_norm: torch.Tensor
    timing_param: torch.Tensor
    x_norm: torch.Tensor
    slice_width_us: torch.Tensor
    sampling_interval_us: torch.Tensor
    modulation_floor: torch.Tensor
    gate_period: torch.Tensor | None = None
    mask_full: torch.Tensor | None = None
    duty_soft: torch.Tensor | None = None
    x_head: torch.Tensor | None = None
    x_template: torch.Tensor | None = None
    x_final: torch.Tensor | None = None
    low_platform_period: torch.Tensor | None = None
    low_platform_full: torch.Tensor | None = None

    def as_physical_tensor(self) -> torch.Tensor:
        return torch.cat(
            [
                self.slice_width_us,
                self.sampling_interval_us,
                self.modulation_floor,
            ],
            dim=1,
        )


def compute_parameter_loss(
    predictions_raw: TwoStagePredictions | GateReconstructionPredictions | DirectRegressionPredictions,
    targets: torch.Tensor,
    jammer_iq: torch.Tensor,
    *,
    jammer_mask: torch.Tensor | None = None,
    scaler: ParameterScaler,
    sample_rate_hz: float,
    jammer_delay_s: float,
    ordering_weight: float,
    consistency_weight: float,
    min_timing_gap_us: float,
    parameterization: str,
    parameter_loss_weights: tuple[float, float, float],
    duty_focus_threshold: float,
    duty_focus_weight: float,
    stage: str,
    mask_reconstruction_weight: float = 1.0,
    gate_tv_weight: float = 0.02,
    plateau_loss_weight: float = 0.0,
    platform_consistency_weight: float = 0.0,
    x_decode_mode: str = "head",
    x_mix_alpha: float = 0.5,
) -> LossOutput:
    if isinstance(predictions_raw, DirectRegressionPredictions):
        return _compute_direct_regression_loss(
            predictions_raw=predictions_raw,
            targets=targets,
            scaler=scaler,
            parameter_loss_weights=parameter_loss_weights,
        )
    if isinstance(predictions_raw, GateReconstructionPredictions):
        return _compute_gate_reconstruction_loss(
            predictions_raw=predictions_raw,
            targets=targets,
            jammer_mask=jammer_mask,
            scaler=scaler,
            sample_rate_hz=sample_rate_hz,
            jammer_delay_s=jammer_delay_s,
            min_timing_gap_us=min_timing_gap_us,
            parameter_loss_weights=parameter_loss_weights,
            mask_reconstruction_weight=mask_reconstruction_weight,
            gate_tv_weight=gate_tv_weight,
            plateau_loss_weight=plateau_loss_weight,
            platform_consistency_weight=platform_consistency_weight,
            x_decode_mode=x_decode_mode,
            x_mix_alpha=x_mix_alpha,
        )
    return _compute_two_stage_parameter_loss(
        predictions_raw=predictions_raw,
        targets=targets,
        jammer_iq=jammer_iq,
        scaler=scaler,
        sample_rate_hz=sample_rate_hz,
        jammer_delay_s=jammer_delay_s,
        ordering_weight=ordering_weight,
        consistency_weight=consistency_weight,
        min_timing_gap_us=min_timing_gap_us,
        parameterization=parameterization,
        parameter_loss_weights=parameter_loss_weights,
        duty_focus_threshold=duty_focus_threshold,
        duty_focus_weight=duty_focus_weight,
        stage=stage,
    )


def decode_predictions(
    predictions_raw: TwoStagePredictions | GateReconstructionPredictions | DirectRegressionPredictions,
    scaler: ParameterScaler,
    min_timing_gap_us: float,
    parameterization: str = "duty",
    sample_rate_hz: float | None = None,
    seq_len: int | None = None,
    jammer_delay_s: float = 0.0,
    x_decode_mode: str = "head",
    x_mix_alpha: float = 0.5,
) -> DecodedPredictions:
    if isinstance(predictions_raw, DirectRegressionPredictions):
        return _decode_direct_regression_predictions(
            predictions_raw=predictions_raw,
            scaler=scaler,
        )
    if isinstance(predictions_raw, GateReconstructionPredictions):
        return _decode_gate_predictions(
            predictions_raw=predictions_raw,
            scaler=scaler,
            min_timing_gap_us=min_timing_gap_us,
            sample_rate_hz=sample_rate_hz,
            seq_len=seq_len,
            jammer_delay_s=jammer_delay_s,
            x_decode_mode=x_decode_mode,
            x_mix_alpha=x_mix_alpha,
        )
    return _decode_two_stage_predictions(
        predictions_raw=predictions_raw,
        scaler=scaler,
        min_timing_gap_us=min_timing_gap_us,
        parameterization=parameterization,
    )


def compute_metrics(
    predictions_raw: TwoStagePredictions | GateReconstructionPredictions | DirectRegressionPredictions,
    targets: torch.Tensor,
    scaler: ParameterScaler,
    min_timing_gap_us: float,
    parameterization: str = "duty",
    x_decode_mode: str = "head",
    x_mix_alpha: float = 0.5,
) -> dict[str, float]:
    predictions = decode_predictions(
        predictions_raw,
        scaler,
        min_timing_gap_us,
        parameterization=parameterization,
        x_decode_mode=x_decode_mode,
        x_mix_alpha=x_mix_alpha,
    ).as_physical_tensor()
    mae = torch.mean(torch.abs(predictions - targets), dim=0)
    rmse = torch.sqrt(torch.mean((predictions - targets) ** 2, dim=0))

    return {
        "slice_width_mae_us": float(mae[0].item()),
        "sampling_interval_mae_us": float(mae[1].item()),
        "modulation_floor_mae": float(mae[2].item()),
        "slice_width_rmse_us": float(rmse[0].item()),
        "sampling_interval_rmse_us": float(rmse[1].item()),
        "modulation_floor_rmse": float(rmse[2].item()),
    }


def _compute_direct_regression_loss(
    predictions_raw: DirectRegressionPredictions,
    targets: torch.Tensor,
    scaler: ParameterScaler,
    parameter_loss_weights: tuple[float, float, float],
) -> LossOutput:
    decoded = _decode_direct_regression_predictions(predictions_raw, scaler)
    physical_predictions = decoded.as_physical_tensor()
    weights = torch.tensor(parameter_loss_weights, dtype=targets.dtype, device=targets.device).view(1, 3)
    regression_terms = F.smooth_l1_loss(physical_predictions, targets, reduction="none") * weights
    regression = regression_terms.mean()
    zero = regression.new_tensor(0.0)
    return LossOutput(total=regression, regression=regression, ordering=zero, consistency=zero)


def build_target_components(
    targets: torch.Tensor,
    scaler: ParameterScaler,
    min_timing_gap_us: float,
    parameterization: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ts_min = scaler.sampling_interval_us.min_value
    ts_max = scaler.sampling_interval_us.max_value
    x_min = scaler.modulation_floor.min_value
    x_max = scaler.modulation_floor.max_value

    slice_width_us = targets[:, 0:1]
    sampling_interval_us = targets[:, 1:2]
    target_ts_norm = (sampling_interval_us - ts_min) / (ts_max - ts_min)
    target_x_norm = (targets[:, 2:3] - x_min) / (x_max - x_min)
    target_duty = slice_width_us / sampling_interval_us.clamp_min(1e-6)

    timing_low, timing_high = _compute_timing_bounds(
        sampling_interval_us,
        scaler,
        min_timing_gap_us,
        parameterization,
    )
    timing_value = _slice_width_to_timing_param(
        slice_width_us,
        sampling_interval_us,
        parameterization,
    )
    target_timing_alpha = (timing_value - timing_low) / (timing_high - timing_low).clamp_min(1e-6)

    return (
        target_ts_norm.clamp(0.0, 1.0),
        target_timing_alpha.clamp(0.0, 1.0),
        target_x_norm.clamp(0.0, 1.0),
        target_duty.clamp(0.0, 1.0),
    )


def build_mask_template(
    predictions: torch.Tensor,
    seq_len: int,
    sample_rate_hz: float,
    jammer_delay_s: float,
    sharpness: float = 18.0,
) -> torch.Tensor:
    time_us = torch.arange(seq_len, device=predictions.device, dtype=predictions.dtype) / sample_rate_hz * 1e6
    relative_us = time_us.unsqueeze(0) - jammer_delay_s * 1e6
    active = torch.sigmoid(relative_us * sharpness)

    slice_width = predictions[:, 0:1]
    sampling_interval = predictions[:, 1:2].clamp_min(slice_width + 1e-3)
    modulation_floor = predictions[:, 2:3].clamp(0.0, 1.0)

    phase = torch.remainder(relative_us.clamp_min(0.0), sampling_interval)
    gate = torch.sigmoid((slice_width - phase) * sharpness) * active
    return modulation_floor + (1.0 - modulation_floor) * gate


def _compute_two_stage_parameter_loss(
    predictions_raw: TwoStagePredictions,
    targets: torch.Tensor,
    jammer_iq: torch.Tensor,
    scaler: ParameterScaler,
    sample_rate_hz: float,
    jammer_delay_s: float,
    ordering_weight: float,
    consistency_weight: float,
    min_timing_gap_us: float,
    parameterization: str,
    parameter_loss_weights: tuple[float, float, float],
    duty_focus_threshold: float,
    duty_focus_weight: float,
    stage: str,
) -> LossOutput:
    decoded = _decode_two_stage_predictions(
        predictions_raw,
        scaler,
        min_timing_gap_us,
        parameterization=parameterization,
    )
    target_ts_norm, target_timing_param, target_x_norm, target_duty = build_target_components(
        targets,
        scaler,
        min_timing_gap_us,
        parameterization=parameterization,
    )

    weights = torch.tensor(parameter_loss_weights, dtype=targets.dtype, device=targets.device).view(1, 3)
    ts_loss = F.smooth_l1_loss(decoded.ts_norm, target_ts_norm, reduction="none")
    timing_loss = F.smooth_l1_loss(decoded.timing_param, target_timing_param, reduction="none")
    x_loss = F.smooth_l1_loss(decoded.x_norm, target_x_norm, reduction="none")

    duty_boost = 1.0 + duty_focus_weight * torch.relu(duty_focus_threshold - target_duty) / max(
        duty_focus_threshold,
        1e-6,
    )
    regression_terms = torch.cat(
        [
            ts_loss * weights[:, 0:1] * duty_boost,
            timing_loss * weights[:, 1:2] * duty_boost,
            x_loss * weights[:, 2:3],
        ],
        dim=1,
    )
    stage_mask = _build_stage_mask(stage, device=targets.device, dtype=targets.dtype)
    regression = (regression_terms * stage_mask).mean()

    if stage == "ts_only":
        zero = regression.new_tensor(0.0)
        return LossOutput(total=regression, regression=regression, ordering=zero, consistency=zero)

    ts_for_shape = decoded.sampling_interval_us.detach() if stage == "duty_x_only" else decoded.sampling_interval_us
    slice_width = decoded.slice_width_us
    ordering = (
        torch.relu(slice_width + min_timing_gap_us - ts_for_shape).mean()
        if ordering_weight > 0
        else regression.new_tensor(0.0)
    )

    consistency = regression.new_tensor(0.0)
    if consistency_weight > 0:
        template_predictions = torch.cat(
            [
                slice_width,
                ts_for_shape,
                decoded.modulation_floor,
            ],
            dim=1,
        )
        template = build_mask_template(
            predictions=template_predictions,
            seq_len=jammer_iq.shape[-1],
            sample_rate_hz=sample_rate_hz,
            jammer_delay_s=jammer_delay_s,
        )
        jammer_envelope = torch.sqrt(torch.sum(jammer_iq**2, dim=1) + 1e-8)
        jammer_envelope = jammer_envelope / jammer_envelope.amax(dim=1, keepdim=True).clamp_min(1e-6)
        consistency = F.smooth_l1_loss(template, jammer_envelope)

    total = regression + ordering_weight * ordering + consistency_weight * consistency
    return LossOutput(total=total, regression=regression, ordering=ordering, consistency=consistency)


def _compute_gate_reconstruction_loss(
    predictions_raw: GateReconstructionPredictions,
    targets: torch.Tensor,
    jammer_mask: torch.Tensor | None,
    scaler: ParameterScaler,
    sample_rate_hz: float,
    jammer_delay_s: float,
    min_timing_gap_us: float,
    parameter_loss_weights: tuple[float, float, float],
    mask_reconstruction_weight: float,
    gate_tv_weight: float,
    plateau_loss_weight: float,
    platform_consistency_weight: float,
    x_decode_mode: str,
    x_mix_alpha: float,
) -> LossOutput:
    if jammer_mask is None:
        raise ValueError("jammer_mask is required for gate reconstruction loss.")

    decoded = _decode_gate_predictions(
        predictions_raw=predictions_raw,
        scaler=scaler,
        min_timing_gap_us=min_timing_gap_us,
        sample_rate_hz=sample_rate_hz,
        seq_len=jammer_mask.shape[-1],
        jammer_delay_s=jammer_delay_s,
        x_decode_mode=x_decode_mode,
        x_mix_alpha=x_mix_alpha,
    )
    physical_predictions = decoded.as_physical_tensor()
    weights = torch.tensor(parameter_loss_weights, dtype=targets.dtype, device=targets.device).view(1, 3)
    regression_terms = F.smooth_l1_loss(physical_predictions, targets, reduction="none") * weights
    regression = regression_terms.mean()

    reconstruction = F.smooth_l1_loss(decoded.mask_full, jammer_mask)
    plateau = decoded.mask_full.new_tensor(0.0)
    if plateau_loss_weight > 0:
        pred_low_mean = _estimate_template_floor_from_mask(decoded.mask_full, decoded.x_head)
        true_low_mean = _estimate_template_floor_from_mask(jammer_mask, targets[:, 2:3])
        plateau = F.smooth_l1_loss(pred_low_mean, true_low_mean)
    platform_consistency = decoded.mask_full.new_tensor(0.0)
    if platform_consistency_weight > 0 and decoded.low_platform_full is not None:
        true_floor = targets[:, 2:3]
        true_gate = ((jammer_mask - true_floor) / (1.0 - true_floor).clamp_min(1e-6)).clamp(0.0, 1.0)
        true_low_weight = 1.0 - true_gate
        platform_consistency = _weighted_smooth_l1(
            decoded.low_platform_full,
            jammer_mask,
            true_low_weight,
        )

    tv = torch.mean(torch.abs(decoded.gate_period[:, 1:] - decoded.gate_period[:, :-1]))
    total = (
        regression
        + mask_reconstruction_weight * reconstruction
        + gate_tv_weight * tv
        + plateau_loss_weight * plateau
        + platform_consistency_weight * platform_consistency
    )
    return LossOutput(total=total, regression=regression, ordering=reconstruction, consistency=tv)


def _decode_two_stage_predictions(
    predictions_raw: TwoStagePredictions,
    scaler: ParameterScaler,
    min_timing_gap_us: float,
    parameterization: str = "duty",
) -> DecodedPredictions:
    ts_norm = predictions_raw.ts_norm.clamp(0.0, 1.0)
    timing_alpha = predictions_raw.timing_param.clamp(0.0, 1.0)
    x_norm = predictions_raw.x_norm.clamp(0.0, 1.0)

    ts_min = scaler.sampling_interval_us.min_value
    ts_max = scaler.sampling_interval_us.max_value
    x_min = scaler.modulation_floor.min_value
    x_max = scaler.modulation_floor.max_value

    sampling_interval_us = ts_norm * (ts_max - ts_min) + ts_min
    timing_low, timing_high = _compute_timing_bounds(
        sampling_interval_us,
        scaler,
        min_timing_gap_us,
        parameterization,
    )
    timing_value = timing_low + timing_alpha * (timing_high - timing_low)
    slice_width_us = _timing_param_to_slice_width(
        timing_value,
        sampling_interval_us,
        parameterization,
    )
    modulation_floor = x_norm * (x_max - x_min) + x_min

    return DecodedPredictions(
        ts_norm=ts_norm,
        timing_param=timing_alpha,
        x_norm=x_norm,
        slice_width_us=slice_width_us,
        sampling_interval_us=sampling_interval_us,
        modulation_floor=modulation_floor,
    )


def _decode_direct_regression_predictions(
    predictions_raw: DirectRegressionPredictions,
    scaler: ParameterScaler,
) -> DecodedPredictions:
    normalized = predictions_raw.normalized_params.clamp(0.0, 1.0)
    physical = scaler.denormalize_tensor(normalized)
    return DecodedPredictions(
        ts_norm=normalized[:, 1:2],
        timing_param=normalized[:, 0:1],
        x_norm=normalized[:, 2:3],
        slice_width_us=physical[:, 0:1],
        sampling_interval_us=physical[:, 1:2],
        modulation_floor=physical[:, 2:3],
    )


def _decode_gate_predictions(
    predictions_raw: GateReconstructionPredictions,
    scaler: ParameterScaler,
    min_timing_gap_us: float,
    sample_rate_hz: float | None = None,
    seq_len: int | None = None,
    jammer_delay_s: float = 0.0,
    x_decode_mode: str = "head",
    x_mix_alpha: float = 0.5,
) -> DecodedPredictions:
    ts_norm = predictions_raw.ts_norm.clamp(0.0, 1.0)
    x_norm = predictions_raw.x_norm.clamp(0.0, 1.0)
    gate_period = predictions_raw.gate_period.clamp(0.0, 1.0)
    low_platform_norm = (
        predictions_raw.low_platform_norm.clamp(0.0, 1.0)
        if predictions_raw.low_platform_norm is not None
        else None
    )

    ts_min = scaler.sampling_interval_us.min_value
    ts_max = scaler.sampling_interval_us.max_value
    x_min = scaler.modulation_floor.min_value
    x_max = scaler.modulation_floor.max_value
    tl_min = scaler.slice_width_us.min_value
    tl_max = scaler.slice_width_us.max_value

    sampling_interval_us = ts_norm * (ts_max - ts_min) + ts_min
    x_head = x_norm * (x_max - x_min) + x_min
    low_platform_period = (
        low_platform_norm * (x_max - x_min) + x_min
        if low_platform_norm is not None
        else None
    )
    duty_soft = torch.mean(gate_period, dim=1, keepdim=True)
    slice_width_us = duty_soft * sampling_interval_us
    slice_width_high = torch.minimum(
        torch.full_like(slice_width_us, tl_max),
        (sampling_interval_us - min_timing_gap_us).clamp_min(tl_min),
    )
    slice_width_us = torch.minimum(torch.maximum(slice_width_us, torch.full_like(slice_width_us, tl_min)), slice_width_high)

    mask_full = None
    x_template = x_head
    modulation_floor = x_head
    low_platform_full = None
    if sample_rate_hz is not None and seq_len is not None:
        gate_full = _interpolate_period_values(
            period_values=gate_period,
            sampling_interval_us=sampling_interval_us,
            seq_len=seq_len,
            sample_rate_hz=sample_rate_hz,
            jammer_delay_s=jammer_delay_s,
        )
        if low_platform_period is not None:
            low_platform_full = _interpolate_period_values(
                period_values=low_platform_period,
                sampling_interval_us=sampling_interval_us,
                seq_len=seq_len,
                sample_rate_hz=sample_rate_hz,
                jammer_delay_s=jammer_delay_s,
            ).clamp(x_min, x_max)
            x_template = _estimate_template_floor_from_period(
                gate_period=gate_period,
                low_platform_period=low_platform_period,
                fallback=x_head,
            ).clamp(x_min, x_max)
        else:
            initial_mask = _build_mask_from_gate_period(
                gate_period=gate_period,
                sampling_interval_us=sampling_interval_us,
                modulation_floor=x_head,
                seq_len=seq_len,
                sample_rate_hz=sample_rate_hz,
                jammer_delay_s=jammer_delay_s,
            )
            x_template = _estimate_template_floor_from_mask(initial_mask, x_head).clamp(x_min, x_max)

        mode = str(x_decode_mode).lower()
        if mode == "template" and low_platform_period is not None:
            modulation_floor = x_template
        elif mode == "template_mix":
            alpha = float(max(0.0, min(1.0, x_mix_alpha)))
            modulation_floor = (alpha * x_head + (1.0 - alpha) * x_template).clamp(x_min, x_max)
        else:
            modulation_floor = x_head
        if low_platform_full is not None and mode == "template":
            mask_full = low_platform_full + (1.0 - low_platform_full) * gate_full
        else:
            mask_full = _build_mask_from_gate_period(
                gate_period=gate_period,
                sampling_interval_us=sampling_interval_us,
                modulation_floor=modulation_floor,
                seq_len=seq_len,
                sample_rate_hz=sample_rate_hz,
                jammer_delay_s=jammer_delay_s,
            )

    return DecodedPredictions(
        ts_norm=ts_norm,
        timing_param=duty_soft,
        x_norm=x_norm,
        slice_width_us=slice_width_us,
        sampling_interval_us=sampling_interval_us,
        modulation_floor=modulation_floor,
        gate_period=gate_period,
        mask_full=mask_full,
        duty_soft=duty_soft,
        x_head=x_head,
        x_template=x_template,
        x_final=modulation_floor,
        low_platform_period=low_platform_period,
        low_platform_full=low_platform_full,
    )


def _build_mask_from_gate_period(
    gate_period: torch.Tensor,
    sampling_interval_us: torch.Tensor,
    modulation_floor: torch.Tensor,
    seq_len: int,
    sample_rate_hz: float,
    jammer_delay_s: float,
) -> torch.Tensor:
    batch_size, gate_bins = gate_period.shape
    dtype = gate_period.dtype
    device = gate_period.device
    time_us = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(0) / sample_rate_hz * 1e6
    relative_us = time_us - jammer_delay_s * 1e6
    phase_norm = torch.remainder(relative_us.clamp_min(0.0), sampling_interval_us.clamp_min(1e-6)) / sampling_interval_us.clamp_min(1e-6)
    position = phase_norm * max(gate_bins - 1, 1)
    index_low = torch.floor(position).long().clamp(0, max(gate_bins - 1, 0))
    index_high = (index_low + 1).clamp(0, max(gate_bins - 1, 0))
    alpha = (position - index_low.to(dtype)).clamp(0.0, 1.0)

    gate_low = torch.gather(gate_period, 1, index_low)
    gate_high = torch.gather(gate_period, 1, index_high)
    interpolated = gate_low + alpha * (gate_high - gate_low)
    return modulation_floor + (1.0 - modulation_floor) * interpolated


def _interpolate_period_values(
    period_values: torch.Tensor,
    sampling_interval_us: torch.Tensor,
    seq_len: int,
    sample_rate_hz: float,
    jammer_delay_s: float,
) -> torch.Tensor:
    batch_size, gate_bins = period_values.shape
    dtype = period_values.dtype
    device = period_values.device
    time_us = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(0) / sample_rate_hz * 1e6
    relative_us = time_us - jammer_delay_s * 1e6
    phase_norm = torch.remainder(relative_us.clamp_min(0.0), sampling_interval_us.clamp_min(1e-6)) / sampling_interval_us.clamp_min(1e-6)
    position = phase_norm * max(gate_bins - 1, 1)
    index_low = torch.floor(position).long().clamp(0, max(gate_bins - 1, 0))
    index_high = (index_low + 1).clamp(0, max(gate_bins - 1, 0))
    alpha = (position - index_low.to(dtype)).clamp(0.0, 1.0)

    value_low = torch.gather(period_values, 1, index_low)
    value_high = torch.gather(period_values, 1, index_high)
    return value_low + alpha * (value_high - value_low)


def _estimate_template_floor_from_mask(
    mask_full: torch.Tensor,
    fallback: torch.Tensor,
    low_fraction: float = 0.25,
    min_candidates: int = 16,
) -> torch.Tensor:
    if mask_full.ndim != 2:
        return fallback

    seq_len = int(mask_full.shape[1])
    if seq_len <= 0:
        return fallback

    k = min(seq_len, max(int(round(seq_len * low_fraction)), min_candidates))
    if k <= 0:
        return fallback

    low_values, _ = torch.topk(mask_full, k=k, dim=1, largest=False, sorted=False)
    return low_values.mean(dim=1, keepdim=True)


def _estimate_template_floor_from_period(
    gate_period: torch.Tensor,
    low_platform_period: torch.Tensor,
    fallback: torch.Tensor,
) -> torch.Tensor:
    low_weight = (1.0 - gate_period).clamp_min(0.0)
    denom = low_weight.sum(dim=1, keepdim=True)
    weighted_mean = (low_platform_period * low_weight).sum(dim=1, keepdim=True) / denom.clamp_min(1e-6)
    use_fallback = denom <= 1e-6
    return torch.where(use_fallback, fallback, weighted_mean)


def _weighted_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    elementwise = F.smooth_l1_loss(prediction, target, reduction="none")
    weighted = elementwise * weight
    return weighted.sum() / weight.sum().clamp_min(1e-6)


def _build_stage_mask(stage: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if stage == "ts_only":
        values = [1.0, 0.0, 0.0]
    elif stage == "duty_x_only":
        values = [0.0, 1.0, 1.0]
    else:
        values = [1.0, 1.0, 1.0]
    return torch.tensor(values, device=device, dtype=dtype).view(1, 3)


def _compute_timing_bounds(
    sampling_interval_us: torch.Tensor,
    scaler: ParameterScaler,
    min_timing_gap_us: float,
    parameterization: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    tl_min = scaler.slice_width_us.min_value
    tl_max = scaler.slice_width_us.max_value

    slice_width_low = torch.full_like(sampling_interval_us, tl_min)
    slice_width_high = torch.minimum(
        torch.full_like(sampling_interval_us, tl_max),
        (sampling_interval_us - min_timing_gap_us).clamp_min(tl_min),
    )
    slice_width_high = torch.maximum(slice_width_high, slice_width_low + 1e-3)

    if parameterization == "duty":
        timing_low = slice_width_low / sampling_interval_us.clamp_min(1e-6)
        timing_high = slice_width_high / sampling_interval_us.clamp_min(1e-6)
    elif parameterization == "gap":
        timing_low = torch.maximum(
            torch.full_like(sampling_interval_us, min_timing_gap_us),
            sampling_interval_us - tl_max,
        )
        timing_high = sampling_interval_us - tl_min
    elif parameterization == "gap_ratio":
        gap_low = torch.maximum(
            torch.full_like(sampling_interval_us, min_timing_gap_us),
            sampling_interval_us - tl_max,
        )
        gap_high = sampling_interval_us - tl_min
        timing_low = gap_low / sampling_interval_us.clamp_min(1e-6)
        timing_high = gap_high / sampling_interval_us.clamp_min(1e-6)
    else:
        raise ValueError(f"Unsupported parameterization '{parameterization}'.")

    timing_high = torch.maximum(timing_high, timing_low + 1e-3)
    return timing_low, timing_high


def _timing_param_to_slice_width(
    timing_param: torch.Tensor,
    sampling_interval_us: torch.Tensor,
    parameterization: str,
) -> torch.Tensor:
    if parameterization == "duty":
        return timing_param * sampling_interval_us
    if parameterization == "gap":
        return sampling_interval_us - timing_param
    if parameterization == "gap_ratio":
        return sampling_interval_us * (1.0 - timing_param)
    raise ValueError(f"Unsupported parameterization '{parameterization}'.")


def _slice_width_to_timing_param(
    slice_width_us: torch.Tensor,
    sampling_interval_us: torch.Tensor,
    parameterization: str,
) -> torch.Tensor:
    if parameterization == "duty":
        return slice_width_us / sampling_interval_us.clamp_min(1e-6)
    if parameterization == "gap":
        return sampling_interval_us - slice_width_us
    if parameterization == "gap_ratio":
        return (sampling_interval_us - slice_width_us) / sampling_interval_us.clamp_min(1e-6)
    raise ValueError(f"Unsupported parameterization '{parameterization}'.")
