from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from confidence import compute_confidence_outputs, summarize_confidence_alignment
from config import ModelConfig, ParameterScaler, build_dataclass
from dataset import NPZISRJDataset, load_manifest
from losses import decode_predictions
from model import build_model
from train import resolve_device


def load_model_bundle(checkpoint_path: str | Path, device: str = "auto"):
    runtime_device = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=runtime_device, weights_only=False)
    scaler = build_dataclass(ParameterScaler, checkpoint["scaler"])
    model = build_model(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(runtime_device)
    model.eval()
    return model, scaler, checkpoint, runtime_device


def predict_from_dataset(checkpoint_path: str | Path, data_path: str | Path, index: int = 0, device: str = "auto") -> dict:
    model, scaler, checkpoint, runtime_device = load_model_bundle(checkpoint_path, device=device)
    dataset = NPZISRJDataset(data_path)
    batch = dataset[index]
    iq = batch["iq"].unsqueeze(0).to(runtime_device)

    with torch.no_grad():
        predictions_raw = model(iq)
        decoded = decode_predictions(
            predictions_raw,
            scaler,
            _resolve_min_timing_gap_us(checkpoint.get("training_config", {})),
            parameterization=_resolve_parameterization(checkpoint.get("training_config", {})),
            sample_rate_hz=float(checkpoint["dataset_config"]["sample_rate_hz"]),
            seq_len=int(batch["jammer_mask"].shape[-1]),
            jammer_delay_s=float(checkpoint["dataset_config"]["jammer_delay_s"]),
            x_decode_mode=_resolve_x_decode_mode(checkpoint),
            x_mix_alpha=_resolve_x_mix_alpha(checkpoint),
        )

    labels = batch["labels"]
    result = {
        "prediction": {
            "slice_width_us": float(decoded.slice_width_us.squeeze(0).item()),
            "sampling_interval_us": float(decoded.sampling_interval_us.squeeze(0).item()),
            "modulation_floor": float(decoded.modulation_floor.squeeze(0).item()),
        },
        "target": {
            "slice_width_us": float(labels[0].item()),
            "sampling_interval_us": float(labels[1].item()),
            "modulation_floor": float(labels[2].item()),
        },
    }
    if decoded.gate_period is not None:
        result["gate_period"] = decoded.gate_period.squeeze(0).cpu().tolist()
    if decoded.mask_full is not None:
        result["mask_full"] = decoded.mask_full.squeeze(0).cpu().tolist()
    if decoded.x_head is not None:
        result["x_head"] = float(decoded.x_head.squeeze(0).item())
    if decoded.x_template is not None:
        result["x_template"] = float(decoded.x_template.squeeze(0).item())
    if decoded.x_final is not None:
        result["x_final"] = float(decoded.x_final.squeeze(0).item())
    return result


def evaluate_dataset(checkpoint_path: str | Path, data_path: str | Path, batch_size: int = 64, device: str = "auto") -> dict:
    model, scaler, checkpoint, runtime_device = load_model_bundle(checkpoint_path, device=device)
    data_path = Path(data_path)
    dataset = NPZISRJDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    manifest = load_manifest(data_path.parent)

    predictions_all = []
    targets_all = []
    x_template_all = []
    for batch in loader:
        iq = batch["iq"].to(runtime_device)
        with torch.no_grad():
            decoded = decode_predictions(
                model(iq),
                scaler,
                _resolve_min_timing_gap_us(checkpoint.get("training_config", {})),
                parameterization=_resolve_parameterization(checkpoint.get("training_config", {})),
                sample_rate_hz=float(checkpoint["dataset_config"]["sample_rate_hz"]),
                seq_len=int(batch["jammer_mask"].shape[-1]),
                jammer_delay_s=float(checkpoint["dataset_config"]["jammer_delay_s"]),
                x_decode_mode=_resolve_x_decode_mode(checkpoint),
                x_mix_alpha=_resolve_x_mix_alpha(checkpoint),
            )
        predictions_all.append(decoded.as_physical_tensor().cpu())
        targets_all.append(batch["labels"].cpu())
        if decoded.x_template is not None:
            x_template_all.append(decoded.x_template.cpu())

    predictions = torch.cat(predictions_all, dim=0).numpy()
    targets_np = torch.cat(targets_all, dim=0).numpy()
    abs_error = np.abs(predictions - targets_np)
    sq_error = (predictions - targets_np) ** 2
    x_template_mae = None
    if x_template_all:
        x_template = torch.cat(x_template_all, dim=0).numpy().reshape(-1)
        x_template_mae = float(np.mean(np.abs(x_template - targets_np[:, 2])))

    hit_tl = abs_error[:, 0] <= 0.15
    hit_ts = abs_error[:, 1] <= 0.25
    hit_x = abs_error[:, 2] <= 0.05
    joint_hit = hit_tl & hit_ts & hit_x
    confidence_outputs = compute_confidence_outputs(
        predictions,
        scaler,
        _resolve_min_timing_gap_us(checkpoint.get("training_config", {})),
    )
    confidence_summary = summarize_confidence_alignment(
        confidence_outputs.confidence_score,
        joint_hit,
    )
    low_jnr_joint_hit_rate = _compute_low_jnr_joint_hit_rate(
        manifest=manifest,
        split_name=data_path.stem,
        count=targets_np.shape[0],
        joint_hit=joint_hit,
    )

    return {
        "num_samples": int(targets_np.shape[0]),
        "x_decode_mode": _resolve_x_decode_mode(checkpoint),
        "x_template_mae": x_template_mae,
        "slice_width": _summarize_error(abs_error[:, 0], sq_error[:, 0], hit_tl),
        "sampling_interval": _summarize_error(abs_error[:, 1], sq_error[:, 1], hit_ts),
        "modulation_floor": _summarize_error(abs_error[:, 2], sq_error[:, 2], hit_x),
        "joint_hit_rate": float(np.mean(joint_hit)),
        "low_jnr_joint_hit_rate": low_jnr_joint_hit_rate,
        "metrics": {
            "slice_width_mae_us": float(np.mean(abs_error[:, 0])),
            "sampling_interval_mae_us": float(np.mean(abs_error[:, 1])),
            "modulation_floor_mae": float(np.mean(abs_error[:, 2])),
            "slice_width_rmse_us": float(np.sqrt(np.mean(sq_error[:, 0]))),
            "sampling_interval_rmse_us": float(np.sqrt(np.mean(sq_error[:, 1]))),
            "modulation_floor_rmse": float(np.sqrt(np.mean(sq_error[:, 2]))),
        },
        "confidence_analysis": {
            "confidence_alignment_score": float(confidence_summary["confidence_alignment_score"]),
            "confidence_bins": confidence_summary["confidence_bins"],
            "mean_confidence": float(np.mean(confidence_outputs.confidence_score)),
            "mean_risk": float(np.mean(confidence_outputs.risk_score)),
            "high_duty_flag_rate": float(np.mean(confidence_outputs.high_duty_flag)),
            "high_x_flag_rate": float(np.mean(confidence_outputs.high_x_flag)),
            "tight_gap_flag_rate": float(np.mean(confidence_outputs.tight_gap_flag)),
            "boundary_flag_rate": float(np.mean(confidence_outputs.boundary_flag)),
        },
    }


def evaluate_all_splits(checkpoint_path: str | Path, data_dir: str | Path, batch_size: int = 64, device: str = "auto") -> dict:
    data_dir = Path(data_dir)
    results = {}
    for split in ("train", "val", "test"):
        split_path = data_dir / f"{split}.npz"
        if split_path.exists():
            results[split] = evaluate_dataset(checkpoint_path, split_path, batch_size=batch_size, device=device)
    return results


def save_evaluation_report(report: dict, output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")


def _summarize_error(abs_error: np.ndarray, sq_error: np.ndarray, hits: np.ndarray) -> dict:
    return {
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean(sq_error))),
        "median_abs_error": float(np.median(abs_error)),
        "p95_abs_error": float(np.quantile(abs_error, 0.95)),
        "hit_rate": float(np.mean(hits)),
    }


def _resolve_min_timing_gap_us(training_config: dict) -> float:
    return float(training_config.get("min_timing_gap_us", training_config.get("gap_min_us", 0.2)))


def _resolve_parameterization(training_config: dict) -> str:
    return str(training_config.get("parameterization", "duty"))


def _resolve_x_decode_mode(checkpoint: dict) -> str:
    model_config = checkpoint.get("model_config", {})
    if isinstance(model_config, dict):
        return str(model_config.get("x_decode_mode", "head"))
    return "head"


def _resolve_x_mix_alpha(checkpoint: dict) -> float:
    model_config = checkpoint.get("model_config", {})
    if isinstance(model_config, dict):
        return float(model_config.get("x_mix_alpha", 0.5))
    return 0.5


def _compute_low_jnr_joint_hit_rate(manifest: dict, split_name: str, count: int, joint_hit: np.ndarray) -> float | None:
    jnr_values = manifest.get("jnr_values_db")
    counts_per_split = manifest.get("jnr_counts_per_split")
    if not jnr_values or not counts_per_split or split_name not in counts_per_split:
        return None
    repeats = int(counts_per_split[split_name])
    if repeats <= 0:
        return None
    ordered = np.repeat(np.array(jnr_values, dtype=np.float32), repeats)[:count]
    low_mask = ordered < 0.0
    if not np.any(low_mask):
        return None
    return float(np.mean(joint_hit[low_mask]))
