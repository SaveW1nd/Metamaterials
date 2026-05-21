from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import ModelConfig
from model import ConvPatchTokenizer, build_model


def test_simple_tokenizer_preserves_output_shapes() -> None:
    tokenizer = ConvPatchTokenizer(
        in_channels=2,
        stem_channels=32,
        hidden_channels=128,
        patch_size=16,
        patch_stride=16,
        variant="simple",
    )
    iq = torch.randn(3, 2, 4000)

    tokens, local_features = tokenizer(iq)

    assert tokens.shape == (3, 250, 128)
    assert local_features.shape == (3, 32, 4000)


def test_gate_reconstruction_builds_with_simple_tokenizer_variant() -> None:
    model = build_model(
        ModelConfig(
            architecture="gate_reconstruction",
            input_channels=2,
            tokenizer_variant="simple",
        )
    )
    iq = torch.randn(2, 2, 4000)

    predictions = model(iq)

    assert predictions.gate_period.shape == (2, 128)


def test_resnet_regression_builds_and_outputs_three_values() -> None:
    model = build_model(
        ModelConfig(
            architecture="resnet_regression",
            input_channels=2,
            stem_channels=32,
            hidden_channels=128,
        )
    )
    iq = torch.randn(2, 2, 4000)

    predictions = model(iq)

    assert predictions.normalized_params.shape == (2, 3)


def test_tcn_regression_builds_and_outputs_three_values() -> None:
    model = build_model(
        ModelConfig(
            architecture="tcn_regression",
            input_channels=2,
            stem_channels=32,
            hidden_channels=128,
        )
    )
    iq = torch.randn(2, 2, 4000)

    predictions = model(iq)

    assert predictions.normalized_params.shape == (2, 3)


def test_densenet_regression_builds_and_outputs_three_values() -> None:
    model = build_model(
        ModelConfig(
            architecture="densenet_regression",
            input_channels=2,
            stem_channels=32,
            hidden_channels=128,
        )
    )
    iq = torch.randn(2, 2, 4000)

    predictions = model(iq)

    assert predictions.normalized_params.shape == (2, 3)
