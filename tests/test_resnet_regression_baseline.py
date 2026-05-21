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
from model import DenseNetRegressionNet, DirectRegressionPredictions, ResNetRegressionNet, TCNRegressionNet


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


def test_resnet_regression_uses_resnet18_style_stage_layout() -> None:
    model = ResNetRegressionNet()

    assert len(model.stage1) == 2
    assert len(model.stage2) == 2
    assert len(model.stage3) == 2
    assert len(model.stage4) == 2
    assert model.head[0].in_features == 512


def test_tcn_regression_uses_four_dilated_temporal_blocks() -> None:
    model = TCNRegressionNet()

    assert len(model.blocks) == 4
    dilations = [block.conv1.dilation[0] for block in model.blocks]
    assert dilations == [1, 2, 4, 8]


def test_densenet_regression_uses_densenet121_style_block_layout() -> None:
    model = DenseNetRegressionNet()

    assert len(model.blocks) == 4
    assert [len(block) for block in model.blocks] == [6, 12, 24, 16]
    assert model.classifier[0].in_features == model.final_num_features
