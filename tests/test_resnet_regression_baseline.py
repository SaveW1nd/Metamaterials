from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import ParameterScaler
from losses import decode_predictions
from model import DirectRegressionPredictions


def test_direct_regression_predictions_decode_to_three_parameters() -> None:
    predictions = DirectRegressionPredictions(
        normalized_params=torch.tensor(
            [
                [0.0, 0.5, 1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
    )

    decoded = decode_predictions(
        predictions_raw=predictions,
        scaler=ParameterScaler(),
        min_timing_gap_us=0.2,
    )

    physical = decoded.as_physical_tensor()
    assert physical.shape == (2, 3)
    assert torch.allclose(physical[0], torch.tensor([0.4, 5.5, 0.5]))
    assert torch.allclose(physical[1], torch.tensor([4.0, 1.0, 0.0]))
