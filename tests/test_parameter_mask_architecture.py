from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import ModelConfig, ParameterScaler
from losses import decode_predictions
from model import ParameterMaskPredictions, build_model


def test_gate_reconstruction_architecture_outputs_physical_parameters() -> None:
    model = build_model(ModelConfig(architecture="gate_reconstruction", input_channels=2))
    iq = torch.randn(2, 2, 4000)

    predictions = model(iq)

    assert isinstance(predictions, ParameterMaskPredictions)
    assert predictions.slice_width_us.shape == (2, 1)
    assert predictions.sampling_interval_us.shape == (2, 1)
    assert predictions.modulation_floor.shape == (2, 1)
    assert torch.all(predictions.slice_width_us > 0)
    assert torch.all(predictions.sampling_interval_us > predictions.slice_width_us)
    assert torch.all((predictions.modulation_floor >= 0.0) & (predictions.modulation_floor <= 1.0))


def test_decode_predictions_builds_mask_from_parameters() -> None:
    predictions = ParameterMaskPredictions(
        slice_width_us=torch.tensor([[2.0]], dtype=torch.float32),
        sampling_interval_us=torch.tensor([[4.0]], dtype=torch.float32),
        modulation_floor=torch.tensor([[0.25]], dtype=torch.float32),
    )

    decoded = decode_predictions(
        predictions,
        ParameterScaler(),
        min_timing_gap_us=0.2,
        sample_rate_hz=1e6,
        seq_len=12,
        jammer_delay_s=0.0,
    )

    assert decoded.mask_full is not None
    assert decoded.mask_full.shape == (1, 12)
    assert torch.all(decoded.mask_full >= 0.25)
    assert torch.all(decoded.mask_full <= 1.0)
    high_region = decoded.mask_full[0, :2]
    low_region = decoded.mask_full[0, 3:4]
    assert torch.mean(high_region) > 0.7
    assert torch.mean(low_region) < 0.6
