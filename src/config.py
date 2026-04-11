from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import torch
import yaml


PRIMARY_PARAM_NAMES = ("slice_width_us", "sampling_interval_us", "modulation_floor")


@dataclass(frozen=True)
class RangeConfig:
    min_value: float
    max_value: float

    def sample(self, rng) -> float:
        return float(rng.uniform(self.min_value, self.max_value))


@dataclass(frozen=True)
class ParameterScaler:
    slice_width_us: RangeConfig = field(default_factory=lambda: RangeConfig(0.4, 4.0))
    sampling_interval_us: RangeConfig = field(default_factory=lambda: RangeConfig(1.0, 10.0))
    modulation_floor: RangeConfig = field(default_factory=lambda: RangeConfig(0.0, 0.5))

    @property
    def mins(self) -> list[float]:
        return [getattr(self, name).min_value for name in PRIMARY_PARAM_NAMES]

    @property
    def maxs(self) -> list[float]:
        return [getattr(self, name).max_value for name in PRIMARY_PARAM_NAMES]

    def normalize_dict(self, params: dict[str, float]) -> list[float]:
        normalized = []
        for name in PRIMARY_PARAM_NAMES:
            value = float(params[name])
            limits = getattr(self, name)
            scale = limits.max_value - limits.min_value
            normalized.append((value - limits.min_value) / scale)
        return normalized

    def denormalize_tensor(self, values: torch.Tensor) -> torch.Tensor:
        mins = torch.tensor(self.mins, dtype=values.dtype, device=values.device)
        maxs = torch.tensor(self.maxs, dtype=values.dtype, device=values.device)
        return values * (maxs - mins) + mins

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass
class SignalConfig:
    bandwidth_hz: float = 40e6
    pulse_width_s: float = 40e-6
    carrier_hz: float = 0.0
    sample_rate_hz: float = 100e6
    num_samples: int = 4000
    target_delay_s: float = 0.0
    jammer_delay_s: float = 0.0
    slice_width_s: float = 2e-6
    sampling_interval_s: float = 4e-6
    modulation_floor: float = 0.5
    snr_db: float = 0.0
    jnr_db: float = 5.0

    def validate(self) -> None:
        if self.slice_width_s <= 0:
            raise ValueError("slice_width_s must be positive.")
        if self.sampling_interval_s <= self.slice_width_s:
            raise ValueError("sampling_interval_s must be larger than slice_width_s.")
        if not 0.0 <= self.modulation_floor <= 1.0:
            raise ValueError("modulation_floor must be in [0, 1].")
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if self.jammer_delay_s < 0:
            raise ValueError("jammer_delay_s must be non-negative.")


@dataclass
class DatasetConfig:
    output_dir: str = "artifacts/demo_dataset"
    train_samples: int = 2048
    val_samples: int = 256
    test_samples: int = 256
    seed: int = 20260325
    sample_rate_hz: float = 100e6
    num_samples: int = 4000
    bandwidth_hz: float = 40e6
    pulse_width_s: float = 40e-6
    carrier_hz: float = 0.0
    target_delay_s: float = 0.0
    jammer_delay_s: float = 0.0
    fixed_snr_db: float | None = None
    snr_db_range: tuple[float, float] = (-10.0, 20.0)
    jnr_db_range: tuple[float, float] = (-6.0, 18.0)
    jnr_discrete_values: list[float] = field(default_factory=list)
    train_samples_per_jnr: int | None = None
    val_samples_per_jnr: int | None = None
    test_samples_per_jnr: int | None = None
    min_interval_gap_us: float = 0.2
    duty_sampling_strategy: str = "balanced_bins"
    duty_bins: list[tuple[float, float]] = field(
        default_factory=lambda: [
            (0.05, 0.20),
            (0.20, 0.35),
            (0.35, 0.55),
            (0.55, 0.85),
        ]
    )
    duty_sampling_attempts: int = 64
    input_representation: str = "feature3"
    use_input_scale: bool = True
    third_channel_mode: str = "mag_log"
    third_channel_smoothing_window: int = 9
    scaler: ParameterScaler = field(default_factory=ParameterScaler)

    def split_sizes(self) -> dict[str, int]:
        if self.uses_discrete_jnr_grid():
            num_points = len(self.jnr_discrete_values)
            return {
                "train": num_points * int(self.train_samples_per_jnr or 0),
                "val": num_points * int(self.val_samples_per_jnr or 0),
                "test": num_points * int(self.test_samples_per_jnr or 0),
            }
        return {
            "train": self.train_samples,
            "val": self.val_samples,
            "test": self.test_samples,
        }

    def uses_discrete_jnr_grid(self) -> bool:
        return bool(self.jnr_discrete_values) and all(
            value is not None and int(value) > 0
            for value in (self.train_samples_per_jnr, self.val_samples_per_jnr, self.test_samples_per_jnr)
        )


@dataclass
class ModelConfig:
    architecture: str = "baseline_transformer"
    input_channels: int = 3
    stem_channels: int = 32
    hidden_channels: int = 128
    ts_embedding_dim: int = 48
    patch_size: int = 16
    patch_stride: int = 16
    ts_pooling: str = "multipool"
    use_local_summary: bool = False
    local_summary_dim: int = 64
    attention_heads: int = 4
    shared_transformer_layers: int = 4
    tx_transformer_layers: int = 2
    feedforward_multiplier: float = 2.0
    dropout: float = 0.1
    gate_bins: int = 128
    gate_representation: str = "single_gate"
    x_decode_mode: str = "head"
    x_mix_alpha: float = 0.5


@dataclass
class TrainingConfig:
    data_dir: str = "artifacts/demo_dataset"
    output_dir: str = "artifacts/checkpoints"
    batch_size: int = 64
    end_to_end_epochs: int = 64
    ts_only_epochs: int = 100
    duty_x_epochs: int = 100
    joint_epochs: int = 100
    learning_rate: float = 3e-4
    scheduler: str = "cosine"
    ts_only_learning_rate: float | None = None
    duty_x_learning_rate: float | None = None
    joint_learning_rate: float | None = None
    ts_only_scheduler: str | None = None
    duty_x_scheduler: str | None = None
    joint_scheduler: str | None = None
    weight_decay: float = 1e-4
    num_workers: int = 0
    device: str = "auto"
    consistency_weight: float = 0.15
    ordering_weight: float = 0.05
    min_timing_gap_us: float = 0.2
    parameterization: str = "duty"
    use_confidence_ranking: bool = False
    checkpoint_tolerance_joint: float = 0.003
    parameter_loss_weights: tuple[float, float, float] = (2.5, 1.0, 1.5)
    mask_reconstruction_weight: float = 1.0
    gate_tv_weight: float = 0.02
    plateau_loss_weight: float = 0.0
    platform_consistency_weight: float = 0.0
    duty_focus_threshold: float = 0.35
    duty_focus_weight: float = 2.0
    seed: int = 20260325
    model: ModelConfig = field(default_factory=ModelConfig)
    scaler: ParameterScaler = field(default_factory=ParameterScaler)


def dataclass_to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {item.name: dataclass_to_dict(getattr(obj, item.name)) for item in fields(obj)}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, list):
        return [dataclass_to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {key: dataclass_to_dict(value) for key, value in obj.items()}
    return obj


def build_dataclass(dataclass_type, overrides: dict[str, Any]):
    kwargs = {}
    for item in fields(dataclass_type):
        if item.name not in overrides:
            continue
        value = overrides[item.name]
        if item.name == "scaler" and isinstance(value, dict):
            kwargs[item.name] = build_dataclass(ParameterScaler, value)
        elif item.name == "model" and isinstance(value, dict):
            kwargs[item.name] = build_dataclass(ModelConfig, value)
        elif dataclass_type is ParameterScaler and isinstance(value, dict):
            kwargs[item.name] = build_dataclass(RangeConfig, value)
        elif isinstance(value, dict) and item.default is not MISSING and is_dataclass(item.default):
            kwargs[item.name] = build_dataclass(type(item.default), value)
        else:
            kwargs[item.name] = value
    return dataclass_type(**kwargs)


def load_yaml_config(path: str | Path, dataclass_type):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return build_dataclass(dataclass_type, raw)


def save_yaml_config(path: str | Path, obj: Any) -> None:
    Path(path).write_text(yaml.safe_dump(dataclass_to_dict(obj), sort_keys=False), encoding="utf-8")
