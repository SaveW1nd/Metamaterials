from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import DatasetConfig
from dataset import _resolve_export_input_scale
from generator import GeneratedSample


def _sample(received_iq: np.ndarray) -> GeneratedSample:
    zeros = np.zeros_like(received_iq, dtype=np.complex64)
    return GeneratedSample(
        received_iq=received_iq.astype(np.complex64),
        target_iq=zeros,
        jammer_iq=zeros,
        jammer_mask=np.zeros(received_iq.shape[0], dtype=np.float32),
        time_axis_s=np.zeros(received_iq.shape[0], dtype=np.float32),
        labels={
            "slice_width_us": 1.0,
            "sampling_interval_us": 2.0,
            "modulation_floor": 0.2,
        },
        signal_config=None,  # type: ignore[arg-type]
    )


def test_resolve_export_input_scale_disabled_returns_one() -> None:
    samples = [
        _sample(np.array([1 + 0j, 2 + 0j, 3 + 0j], dtype=np.complex64)),
        _sample(np.array([2 + 0j, 4 + 0j, 8 + 0j], dtype=np.complex64)),
    ]

    config = DatasetConfig(use_input_scale=False)

    scale = _resolve_export_input_scale(samples, config)

    assert scale == 1.0


def test_resolve_export_input_scale_enabled_uses_dataset_statistic() -> None:
    samples = [
        _sample(np.array([1 + 0j, 2 + 0j, 3 + 0j], dtype=np.complex64)),
        _sample(np.array([2 + 0j, 4 + 0j, 8 + 0j], dtype=np.complex64)),
    ]

    config = DatasetConfig(use_input_scale=True)

    scale = _resolve_export_input_scale(samples, config)

    expected = float(np.median([np.percentile([1.0, 2.0, 3.0], 99), np.percentile([2.0, 4.0, 8.0], 99)]))
    assert np.isclose(scale, expected)
