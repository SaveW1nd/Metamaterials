from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from config import DatasetConfig, PRIMARY_PARAM_NAMES, ParameterScaler, dataclass_to_dict
from generator import (
    GeneratedSample,
    build_dataset_arrays,
    build_model_input,
    estimate_input_scale,
    generate_isrj_sample_with_rng,
    sample_signal_config,
)


class SyntheticISRJDataset(Dataset):
    def __init__(
        self,
        dataset_config: DatasetConfig,
        split: str,
        sample_count: int,
        seed_offset: int = 0,
        input_scale: float = 1.0,
    ):
        self.dataset_config = dataset_config
        self.split = split
        self.sample_count = sample_count
        self.seed_offset = seed_offset
        self.input_scale = max(float(input_scale), 1e-6)

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.dataset_config.seed + self.seed_offset + index)
        signal_config = sample_signal_config(rng, self.dataset_config)
        sample = generate_isrj_sample_with_rng(signal_config, rng)
        return sample_to_tensors(
            sample,
            self.dataset_config.scaler,
            self.input_scale,
            input_representation=self.dataset_config.input_representation,
            third_channel_mode=self.dataset_config.third_channel_mode,
            smoothing_window=self.dataset_config.third_channel_smoothing_window,
        )


class NPZISRJDataset(Dataset):
    def __init__(self, npz_path: str | Path):
        bundle = np.load(npz_path)
        self.iq = bundle["iq"].astype(np.float32)
        self.jammer_iq = bundle["jammer_iq"].astype(np.float32)
        self.jammer_mask = bundle["jammer_mask"].astype(np.float32)
        self.labels = bundle["labels"].astype(np.float32)
        self.labels_norm = bundle["labels_norm"].astype(np.float32)

    def __len__(self) -> int:
        return int(self.iq.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "iq": torch.from_numpy(self.iq[index]),
            "jammer_iq": torch.from_numpy(self.jammer_iq[index]),
            "jammer_mask": torch.from_numpy(self.jammer_mask[index]),
            "labels": torch.from_numpy(self.labels[index]),
            "labels_norm": torch.from_numpy(self.labels_norm[index]),
        }


def sample_to_tensors(
    sample: GeneratedSample,
    scaler: ParameterScaler,
    input_scale: float,
    input_representation: str = "feature3",
    third_channel_mode: str = "mag_log",
    smoothing_window: int = 9,
) -> dict[str, torch.Tensor]:
    labels = np.array([sample.labels[name] for name in PRIMARY_PARAM_NAMES], dtype=np.float32)
    labels_norm = np.array(scaler.normalize_dict(sample.labels), dtype=np.float32)
    return {
        "iq": torch.from_numpy(
            build_model_input(
                sample.received_iq,
                input_scale,
                input_representation=input_representation,
                third_channel_mode=third_channel_mode,
                smoothing_window=smoothing_window,
            )
        ),
        "jammer_iq": torch.from_numpy(
            np.stack([sample.jammer_iq.real, sample.jammer_iq.imag], axis=0).astype(np.float32)
        ),
        "jammer_mask": torch.from_numpy(sample.jammer_mask.astype(np.float32)),
        "labels": torch.from_numpy(labels),
        "labels_norm": torch.from_numpy(labels_norm),
    }


def export_dataset(dataset_config: DatasetConfig) -> Path:
    output_dir = Path(dataset_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    offset = 0
    split_sizes = dataset_config.split_sizes()
    train_samples = _generate_split_samples(dataset_config, "train", split_sizes["train"], offset)
    input_scale = _resolve_export_input_scale(train_samples, dataset_config)
    _write_split(output_dir, "train", train_samples, dataset_config, input_scale)
    offset += split_sizes["train"] * 13

    for split in ("val", "test"):
        sample_count = split_sizes[split]
        split_samples = _generate_split_samples(dataset_config, split, sample_count, offset)
        _write_split(output_dir, split, split_samples, dataset_config, input_scale)
        offset += sample_count * 13

    manifest = {
        "primary_param_names": list(PRIMARY_PARAM_NAMES),
        "input_scale": input_scale,
        "dataset_config": dataclass_to_dict(dataset_config),
    }
    if dataset_config.uses_discrete_jnr_grid():
        manifest["jnr_values_db"] = [float(value) for value in dataset_config.jnr_discrete_values]
        manifest["jnr_counts_per_split"] = {
            "train": int(dataset_config.train_samples_per_jnr or 0),
            "val": int(dataset_config.val_samples_per_jnr or 0),
            "test": int(dataset_config.test_samples_per_jnr or 0),
        }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_dir


def load_manifest(data_dir: str | Path) -> dict:
    return json.loads((Path(data_dir) / "manifest.json").read_text(encoding="utf-8"))


def _resolve_export_input_scale(samples: list[GeneratedSample], dataset_config: DatasetConfig) -> float:
    if not dataset_config.use_input_scale:
        return 1.0
    return estimate_input_scale(samples)


def _generate_split_samples(
    dataset_config: DatasetConfig,
    split: str,
    sample_count: int,
    offset: int,
) -> list[GeneratedSample]:
    if dataset_config.uses_discrete_jnr_grid():
        return _generate_discrete_jnr_split_samples(dataset_config, split, offset)

    samples: list[GeneratedSample] = []
    for local_index in range(sample_count):
        rng = np.random.default_rng(dataset_config.seed + offset + local_index)
        signal_config = sample_signal_config(rng, dataset_config)
        samples.append(generate_isrj_sample_with_rng(signal_config, rng))
    return samples


def _generate_discrete_jnr_split_samples(
    dataset_config: DatasetConfig,
    split: str,
    offset: int,
) -> list[GeneratedSample]:
    samples: list[GeneratedSample] = []
    repeats = _samples_per_jnr_for_split(dataset_config, split)
    local_index = 0
    for jnr_db in dataset_config.jnr_discrete_values:
        for _ in range(repeats):
            rng = np.random.default_rng(dataset_config.seed + offset + local_index)
            signal_config = sample_signal_config(rng, dataset_config)
            signal_config.jnr_db = float(jnr_db)
            samples.append(generate_isrj_sample_with_rng(signal_config, rng))
            local_index += 1
    return samples


def _samples_per_jnr_for_split(dataset_config: DatasetConfig, split: str) -> int:
    if split == "train":
        return int(dataset_config.train_samples_per_jnr or 0)
    if split == "val":
        return int(dataset_config.val_samples_per_jnr or 0)
    if split == "test":
        return int(dataset_config.test_samples_per_jnr or 0)
    raise ValueError(f"Unsupported split: {split}")


def _write_split(
    output_dir: Path,
    split: str,
    samples: list[GeneratedSample],
    dataset_config: DatasetConfig,
    input_scale: float,
) -> None:
    arrays = build_dataset_arrays(
        samples,
        input_scale,
        input_representation=dataset_config.input_representation,
        third_channel_mode=dataset_config.third_channel_mode,
        smoothing_window=dataset_config.third_channel_smoothing_window,
    )
    labels_norm = np.stack(
        [dataset_config.scaler.normalize_dict(sample.labels) for sample in samples],
        axis=0,
    ).astype(np.float32)
    np.savez_compressed(
        output_dir / f"{split}.npz",
        iq=arrays["iq"],
        jammer_iq=arrays["jammer_iq"],
        jammer_mask=arrays["jammer_mask"],
        labels=arrays["labels"],
        labels_norm=labels_norm,
    )
