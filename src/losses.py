from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from config import ParameterScaler
from model import ParameterMaskPredictions, TwoStagePredictions


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
    mask_full: torch.Tensor | None = None

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
    predictions_raw: TwoStagePredictions | ParameterMaskPredictions,
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
) -> LossOutput:
    if isinstance(predictions_raw, ParameterMaskPredictions):
        return _compute_parameter_mask_loss(
            predictions_raw=predictions_raw,
            targets=targets,
            jammer_mask=jammer_mask,
            scaler=scaler,
            sample_rate_hz=sample_rate_hz,
            jammer_delay_s=jammer_delay_s,
            parameter_loss_weights=parameter_loss_weights,
            mask_reconstruction_weight=mask_reconstruction_weight,
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
    predictions_raw: TwoStagePredictions | ParameterMaskPredictions,
    scaler: ParameterScaler,
    min_timing_gap_us: float,
    parameterization: str = "duty",
    sample_rate_hz: float | None = None,
    seq_len: int | None = None,
    jammer_delay_s: float = 0.0,
) -> DecodedPredictions:
    if isinstance(predictions_raw, ParameterMaskPredictions):
        return _decode_parameter_mask_predictions(
            predictions_raw=predictions_raw,
            scaler=scaler,
            sample_rate_hz=sample_rate_hz,
            seq_len=seq_len,
            jammer_delay_s=jammer_delay_s,
        )
    return _decode_two_stage_predictions(
        predictions_raw=predictions_raw,
        scaler=scaler,
        min_timing_gap_us=min_timing_gap_us,
        parameterization=parameterization,
    )


def compute_metrics(
    predictions_raw: TwoStagePredictions | ParameterMaskPredictions,
    targets: torch.Tensor,
    scaler: ParameterScaler,
    min_timing_gap_us: float,
    parameterization: str = "duty",
) -> dict[str, float]:
    predictions = decode_predictions(
        predictions_raw,
        scaler,
        min_timing_gap_us,
        parameterization=parameterization,
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


def _compute_parameter_mask_loss(
    predictions_raw: ParameterMaskPredictions,
    targets: torch.Tensor,
    jammer_mask: torch.Tensor | None,
    scaler: ParameterScaler,
    sample_rate_hz: float,
    jammer_delay_s: float,
    parameter_loss_weights: tuple[float, float, float],
    mask_reconstruction_weight: float,
) -> LossOutput:
    if jammer_mask is None:
        raise ValueError("jammer_mask is required for parameter-mask loss.")

    decoded = _decode_parameter_mask_predictions(
        predictions_raw=predictions_raw,
        scaler=scaler,
        sample_rate_hz=sample_rate_hz,
        seq_len=jammer_mask.shape[-1],
        jammer_delay_s=jammer_delay_s,
    )
    physical_predictions = decoded.as_physical_tensor()
    weights = torch.tensor(parameter_loss_weights, dtype=targets.dtype, device=targets.device).view(1, 3)
    regression_terms = F.smooth_l1_loss(physical_predictions, targets, reduction="none") * weights
    regression = regression_terms.mean()

    reconstruction = F.smooth_l1_loss(decoded.mask_full, jammer_mask)
    total = regression + mask_reconstruction_weight * reconstruction
    zero = regression.new_tensor(0.0)
    return LossOutput(total=total, regression=regression, ordering=reconstruction, consistency=zero)


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


def _decode_parameter_mask_predictions(
    predictions_raw: ParameterMaskPredictions,
    scaler: ParameterScaler,
    sample_rate_hz: float | None = None,
    seq_len: int | None = None,
    jammer_delay_s: float = 0.0,
) -> DecodedPredictions:
    slice_width_us = predictions_raw.slice_width_us.clamp_min(1e-6)
    sampling_interval_us = torch.maximum(
        predictions_raw.sampling_interval_us,
        slice_width_us + 1e-6,
    )
    modulation_floor = predictions_raw.modulation_floor.clamp(0.0, 1.0)
    mask_full = None
    if sample_rate_hz is not None and seq_len is not None:
        mask_full = _build_mask_from_parameters(
            slice_width_us=slice_width_us,
            sampling_interval_us=sampling_interval_us,
            modulation_floor=modulation_floor,
            seq_len=seq_len,
            sample_rate_hz=sample_rate_hz,
            jammer_delay_s=jammer_delay_s,
        )

    return DecodedPredictions(
        ts_norm=torch.zeros_like(slice_width_us),
        timing_param=torch.zeros_like(slice_width_us),
        x_norm=modulation_floor,
        slice_width_us=slice_width_us,
        sampling_interval_us=sampling_interval_us,
        modulation_floor=modulation_floor,
        mask_full=mask_full,
    )


def _build_mask_from_parameters(
    slice_width_us: torch.Tensor,
    sampling_interval_us: torch.Tensor,
    modulation_floor: torch.Tensor,
    seq_len: int,
    sample_rate_hz: float,
    jammer_delay_s: float,
    sharpness: float = 18.0,
) -> torch.Tensor:
    dtype = slice_width_us.dtype
    device = slice_width_us.device
    time_us = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(0) / sample_rate_hz * 1e6
    relative_us = time_us - jammer_delay_s * 1e6
    phase_norm = torch.remainder(relative_us.clamp_min(0.0), sampling_interval_us.clamp_min(1e-6)) / sampling_interval_us.clamp_min(1e-6)
    duty_ratio = slice_width_us / sampling_interval_us.clamp_min(1e-6)
    active = torch.sigmoid(relative_us * sharpness)
    soft_gate = torch.sigmoid((duty_ratio - phase_norm) * sharpness) * active
    return modulation_floor + (1.0 - modulation_floor) * soft_gate


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
