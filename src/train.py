from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from config import ModelConfig, ParameterScaler, TrainingConfig, build_dataclass, dataclass_to_dict
from confidence import compute_confidence_outputs, summarize_confidence_alignment
from dataset import NPZISRJDataset, load_manifest
from losses import compute_parameter_loss, decode_predictions
from model import PGIQNet, build_model


def train_model(config: TrainingConfig) -> Path:
    _set_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(config.data_dir)
    dataset_config = manifest["dataset_config"]
    input_scale = float(manifest.get("input_scale", 1.0))
    scaler = build_dataclass(ParameterScaler, dataset_config["scaler"])

    train_loader = DataLoader(
        NPZISRJDataset(Path(config.data_dir) / "train.npz"),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        NPZISRJDataset(Path(config.data_dir) / "val.npz"),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = build_model(config.model).to(device)
    stages = _build_training_stages(config)

    best_score: dict[str, float] | float = float("inf")
    history: list[dict[str, float | str]] = []
    best_path = output_dir / "best_model.pt"
    global_epoch = 0

    for stage in stages:
        stage_name = str(stage["name"])
        stage_epochs = int(stage["epochs"])
        _set_stage_trainability(model, stage_name)
        stage_learning_rate = _resolve_stage_learning_rate(config, stage_name)
        optimizer = AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=stage_learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = _build_stage_scheduler(config, stage_name, optimizer, stage_epochs)

        for stage_epoch in range(1, stage_epochs + 1):
            global_epoch += 1
            train_stats = _run_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                dataset_config=dataset_config,
                training_config=config,
                stage=stage_name,
                split_name="train",
                train_mode=True,
            )
            if scheduler is not None:
                scheduler.step()
            val_stats = _run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                device=device,
                scaler=scaler,
                dataset_config=dataset_config,
                training_config=config,
                stage=stage_name,
                split_name="val",
                train_mode=False,
            )

            epoch_record = {
                "epoch": global_epoch,
                "stage": stage_name,
                "stage_epoch": stage_epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **_prefix_dict("train_", train_stats),
                **_prefix_dict("val_", val_stats),
            }
            history.append(epoch_record)
            print(
                f"stage={stage_name} epoch={stage_epoch}/{stage_epochs} "
                f"lr={optimizer.param_groups[0]['lr']:.6f} "
                f"train_loss={train_stats['loss']:.4f} val_loss={val_stats['loss']:.4f} "
                f"val_joint={val_stats['joint_hit_rate']:.4f} "
                f"val_tl_mae={val_stats['slice_width_mae_us']:.4f}us "
                f"val_ts_mae={val_stats['sampling_interval_mae_us']:.4f}us "
                f"val_x_mae={val_stats['modulation_floor_mae']:.4f}"
            )

            if _should_update_checkpoint(best_score, val_stats, config):
                best_score = val_stats
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_config": dataclass_to_dict(config.model),
                        "training_config": dataclass_to_dict(config),
                        "scaler": scaler.to_dict(),
                        "dataset_config": dataset_config,
                        "input_scale": input_scale,
                        "best_stage": stage_name,
                        "history": history,
                    },
                    best_path,
                )

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return best_path


def evaluate_checkpoint(checkpoint_path: str | Path, data_dir: str | Path, batch_size: int = 64) -> dict[str, float]:
    device = resolve_device("auto")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    scaler = build_dataclass(ParameterScaler, checkpoint["scaler"])
    dataset_config = checkpoint["dataset_config"]
    training_config = checkpoint.get("training_config", {})
    stage = str(checkpoint.get("best_stage", "joint"))
    return _evaluate_checkpoint_model(
        checkpoint=checkpoint,
        model_config=checkpoint["model_config"],
        scaler=scaler,
        dataset_config=dataset_config,
        training_config=training_config,
        data_dir=data_dir,
        batch_size=batch_size,
        device=device,
        stage=stage,
    )


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)


def _evaluate_checkpoint_model(
    checkpoint: dict,
    model_config: dict,
    scaler: ParameterScaler,
    dataset_config: dict,
    training_config: dict,
    data_dir: str | Path,
    batch_size: int,
    device: torch.device,
    stage: str,
) -> dict[str, float]:
    model = build_model(ModelConfig(**model_config))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    test_loader = DataLoader(NPZISRJDataset(Path(data_dir) / "test.npz"), batch_size=batch_size, shuffle=False)
    return _run_epoch(
        model=model,
        loader=test_loader,
        optimizer=None,
        device=device,
        scaler=scaler,
        dataset_config=dataset_config,
        training_config=_training_config_from_checkpoint(training_config),
        stage=stage,
        split_name="test",
        train_mode=False,
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    scaler: ParameterScaler,
    dataset_config: dict,
    training_config: TrainingConfig,
    stage: str,
    split_name: str,
    train_mode: bool,
) -> dict[str, float]:
    if train_mode:
        _set_module_modes(model, stage)
    else:
        model.eval()

    total_loss = 0.0
    predictions_all: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []
    jnr_all: list[torch.Tensor] = []
    sample_cursor = 0

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for batch in loader:
            iq = batch["iq"].to(device)
            jammer_iq = batch["jammer_iq"].to(device)
            jammer_mask = batch["jammer_mask"].to(device)
            labels = batch["labels"].to(device)

            predictions = model(iq)
            loss_output = compute_parameter_loss(
                predictions_raw=predictions,
                targets=labels,
                jammer_iq=jammer_iq,
                jammer_mask=jammer_mask,
                scaler=scaler,
                sample_rate_hz=float(dataset_config["sample_rate_hz"]),
                jammer_delay_s=float(dataset_config["jammer_delay_s"]),
                ordering_weight=training_config.ordering_weight,
                consistency_weight=training_config.consistency_weight,
                min_timing_gap_us=_resolve_min_timing_gap_us(training_config),
                parameterization=training_config.parameterization,
                parameter_loss_weights=training_config.parameter_loss_weights,
                duty_focus_threshold=training_config.duty_focus_threshold,
                duty_focus_weight=training_config.duty_focus_weight,
                stage=stage,
                mask_reconstruction_weight=training_config.mask_reconstruction_weight,
                gate_tv_weight=training_config.gate_tv_weight,
                plateau_loss_weight=training_config.plateau_loss_weight,
                platform_consistency_weight=training_config.platform_consistency_weight,
                x_decode_mode=_resolve_x_decode_mode(training_config),
                x_mix_alpha=_resolve_x_mix_alpha(training_config),
            )

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss_output.total.backward()
                optimizer.step()

            decoded = decode_predictions(
                predictions,
                scaler,
                _resolve_min_timing_gap_us(training_config),
                parameterization=training_config.parameterization,
                sample_rate_hz=float(dataset_config["sample_rate_hz"]),
                seq_len=jammer_mask.shape[-1],
                jammer_delay_s=float(dataset_config["jammer_delay_s"]),
                x_decode_mode=_resolve_x_decode_mode(training_config),
                x_mix_alpha=_resolve_x_mix_alpha(training_config),
            )
            total_loss += float(loss_output.total.item()) * iq.shape[0]
            predictions_all.append(decoded.as_physical_tensor().detach().cpu())
            targets_all.append(labels.detach().cpu())
            if "jnr_db" in batch:
                jnr_all.append(batch["jnr_db"].detach().cpu())
            elif not train_mode:
                jnr_chunk = _recover_jnr_chunk(
                    dataset_config=dataset_config,
                    split_name=split_name,
                    start=sample_cursor,
                    count=iq.shape[0],
                )
                if jnr_chunk is not None:
                    jnr_all.append(torch.from_numpy(jnr_chunk))
            sample_cursor += iq.shape[0]

    predictions_np = torch.cat(predictions_all, dim=0).numpy()
    targets_np = torch.cat(targets_all, dim=0).numpy()
    abs_error = np.abs(predictions_np - targets_np)
    sq_error = (predictions_np - targets_np) ** 2

    hit_tl = abs_error[:, 0] <= 0.15
    hit_ts = abs_error[:, 1] <= 0.25
    hit_x = abs_error[:, 2] <= 0.05
    joint_hit = hit_tl & hit_ts & hit_x

    metrics: dict[str, float] = {
        "slice_width_mae_us": float(np.mean(abs_error[:, 0])),
        "sampling_interval_mae_us": float(np.mean(abs_error[:, 1])),
        "modulation_floor_mae": float(np.mean(abs_error[:, 2])),
        "slice_width_rmse_us": float(np.sqrt(np.mean(sq_error[:, 0]))),
        "sampling_interval_rmse_us": float(np.sqrt(np.mean(sq_error[:, 1]))),
        "modulation_floor_rmse": float(np.sqrt(np.mean(sq_error[:, 2]))),
        "loss": total_loss / len(loader.dataset),
        "slice_width_hit_rate": float(np.mean(hit_tl)),
        "sampling_interval_hit_rate": float(np.mean(hit_ts)),
        "modulation_floor_hit_rate": float(np.mean(hit_x)),
        "joint_hit_rate": float(np.mean(joint_hit)),
    }
    metrics["weighted_mae"] = float(
        2.0 * metrics["sampling_interval_mae_us"]
        + metrics["slice_width_mae_us"]
        + metrics["modulation_floor_mae"]
    )
    metrics["x_decode_mode"] = _resolve_x_decode_mode(training_config)

    confidence_outputs = compute_confidence_outputs(
        predictions_np,
        scaler,
        _resolve_min_timing_gap_us(training_config),
    )
    confidence_summary = summarize_confidence_alignment(
        confidence_outputs.confidence_score,
        joint_hit,
    )
    metrics["confidence_alignment_score"] = float(confidence_summary["confidence_alignment_score"])
    for bin_summary in confidence_summary["confidence_bins"]:
        label = str(bin_summary["label"])
        metrics[f"confidence_{label}_count"] = float(bin_summary["count"])
        metrics[f"confidence_{label}_mean"] = float(bin_summary["mean_confidence"])
        metrics[f"confidence_{label}_joint_hit_rate"] = float(bin_summary["joint_hit_rate"])
    low_jnr_joint = _compute_low_jnr_joint_hit(jnr_all, joint_hit)
    if low_jnr_joint is not None:
        metrics["low_jnr_joint_hit_rate"] = float(low_jnr_joint)
    return metrics


def _build_training_stages(config: TrainingConfig) -> list[dict[str, int | str]]:
    if str(config.model.architecture).lower() in {"gate_reconstruction", "resnet_regression"}:
        return [{"name": "end_to_end", "epochs": int(config.end_to_end_epochs)}]
    stages = [
        {"name": "ts_only", "epochs": config.ts_only_epochs},
        {"name": "duty_x_only", "epochs": config.duty_x_epochs},
        {"name": "joint", "epochs": config.joint_epochs},
    ]
    return [stage for stage in stages if int(stage["epochs"]) > 0]


def _resolve_stage_learning_rate(config: TrainingConfig, stage: str) -> float:
    if stage == "ts_only" and config.ts_only_learning_rate is not None:
        return float(config.ts_only_learning_rate)
    if stage == "duty_x_only" and config.duty_x_learning_rate is not None:
        return float(config.duty_x_learning_rate)
    if stage == "joint" and config.joint_learning_rate is not None:
        return float(config.joint_learning_rate)
    return float(config.learning_rate)


def _build_stage_scheduler(
    config: TrainingConfig,
    stage: str,
    optimizer: AdamW,
    stage_epochs: int,
):
    scheduler_name = _resolve_stage_scheduler_name(config, stage)
    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max(stage_epochs, 1))
    raise ValueError(f"Unsupported scheduler '{scheduler_name}' for stage '{stage}'.")


def _resolve_stage_scheduler_name(config: TrainingConfig, stage: str) -> str:
    if stage == "ts_only" and config.ts_only_scheduler is not None:
        return str(config.ts_only_scheduler)
    if stage == "duty_x_only" and config.duty_x_scheduler is not None:
        return str(config.duty_x_scheduler)
    if stage == "joint" and config.joint_scheduler is not None:
        return str(config.joint_scheduler)
    return str(config.scheduler)


def _set_stage_trainability(model: nn.Module, stage: str) -> None:
    if not isinstance(model, PGIQNet):
        for parameter in model.parameters():
            parameter.requires_grad = True
        return

    _set_requires_grad(model.shared_modules(), False)
    _set_requires_grad(model.ts_modules(), False)
    _set_requires_grad(model.tx_modules(), False)

    if stage == "ts_only":
        _set_requires_grad(model.shared_modules(), True)
        _set_requires_grad(model.ts_modules(), True)
    elif stage == "duty_x_only":
        _set_requires_grad(model.tx_modules(), True)
    else:
        _set_requires_grad(model.shared_modules(), True)
        _set_requires_grad(model.ts_modules(), True)
        _set_requires_grad(model.tx_modules(), True)


def _set_module_modes(model: nn.Module, stage: str) -> None:
    if not isinstance(model, PGIQNet):
        model.train()
        return

    model.eval()
    if stage == "ts_only":
        _set_modules_train(model.shared_modules() + model.ts_modules())
    elif stage == "duty_x_only":
        _set_modules_train(model.tx_modules())
    else:
        model.train()


def _set_requires_grad(modules: list[nn.Module], value: bool) -> None:
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = value


def _set_modules_train(modules: list[nn.Module]) -> None:
    for module in modules:
        module.train()


def _should_update_checkpoint(
    best_metrics: dict[str, float] | float,
    current_metrics: dict[str, float],
    config: TrainingConfig,
) -> bool:
    if isinstance(best_metrics, float):
        return True

    joint_tolerance = float(config.checkpoint_tolerance_joint)
    current_joint = float(current_metrics["joint_hit_rate"])
    best_joint = float(best_metrics["joint_hit_rate"])
    if current_joint > best_joint + joint_tolerance:
        return True
    if current_joint < best_joint - joint_tolerance:
        return False

    if config.use_confidence_ranking:
        current_alignment = float(current_metrics.get("confidence_alignment_score", 0.0))
        best_alignment = float(best_metrics.get("confidence_alignment_score", 0.0))
        if current_alignment > best_alignment + 1e-6:
            return True
        if current_alignment < best_alignment - 1e-6:
            return False

    return float(current_metrics["weighted_mae"]) < float(best_metrics["weighted_mae"]) - 1e-6


def _resolve_min_timing_gap_us(training_config: TrainingConfig | dict) -> float:
    if isinstance(training_config, dict):
        return float(training_config.get("min_timing_gap_us", training_config.get("gap_min_us", 0.2)))
    return float(training_config.min_timing_gap_us)


def _resolve_x_decode_mode(training_config: TrainingConfig | dict) -> str:
    if isinstance(training_config, dict):
        model_config = training_config.get("model", {})
        if isinstance(model_config, dict):
            return str(model_config.get("x_decode_mode", "head"))
        return "head"
    return str(training_config.model.x_decode_mode)


def _resolve_x_mix_alpha(training_config: TrainingConfig | dict) -> float:
    if isinstance(training_config, dict):
        model_config = training_config.get("model", {})
        if isinstance(model_config, dict):
            return float(model_config.get("x_mix_alpha", 0.5))
        return 0.5
    return float(training_config.model.x_mix_alpha)


def _training_config_from_checkpoint(training_config: dict) -> TrainingConfig:
    if isinstance(training_config, TrainingConfig):
        return training_config
    normalized = dict(training_config)
    if "parameter_loss_weights" not in normalized and "regression_weights" in normalized:
        normalized["parameter_loss_weights"] = normalized["regression_weights"]
    if "min_timing_gap_us" not in normalized and "gap_min_us" in normalized:
        normalized["min_timing_gap_us"] = normalized["gap_min_us"]
    return build_dataclass(TrainingConfig, normalized)


def _recover_jnr_chunk(
    dataset_config: dict,
    split_name: str,
    start: int,
    count: int,
) -> np.ndarray | None:
    jnr_values = dataset_config.get("jnr_discrete_values", [])
    if not jnr_values:
        return None

    if split_name == "train":
        repeats = int(dataset_config.get("train_samples_per_jnr") or 0)
    elif split_name == "val":
        repeats = int(dataset_config.get("val_samples_per_jnr") or 0)
    elif split_name == "test":
        repeats = int(dataset_config.get("test_samples_per_jnr") or 0)
    else:
        return None

    if repeats <= 0:
        return None

    ordered = np.repeat(np.array(jnr_values, dtype=np.float32), repeats)
    return ordered[start : start + count]


def _compute_low_jnr_joint_hit(jnr_all: list[torch.Tensor], joint_hit: np.ndarray) -> float | None:
    if not jnr_all:
        return None
    jnr_values = torch.cat(jnr_all, dim=0).cpu().numpy().reshape(-1)
    if jnr_values.shape[0] != joint_hit.shape[0]:
        return None
    low_mask = jnr_values < 0.0
    if not np.any(low_mask):
        return None
    return float(np.mean(joint_hit[low_mask]))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prefix_dict(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}{key}": value for key, value in values.items()}
