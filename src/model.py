from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from config import ModelConfig


@dataclass
class TwoStagePredictions:
    ts_norm: torch.Tensor
    timing_param: torch.Tensor
    x_norm: torch.Tensor
    ts_embedding: torch.Tensor

    def as_tensor(self) -> torch.Tensor:
        return torch.cat([self.ts_norm, self.timing_param, self.x_norm], dim=1)


@dataclass
class GateReconstructionPredictions:
    ts_norm: torch.Tensor
    x_norm: torch.Tensor
    gate_period: torch.Tensor
    low_platform_norm: torch.Tensor | None = None

    def as_tensor(self) -> torch.Tensor:
        tensors = [self.ts_norm, self.x_norm, self.gate_period]
        if self.low_platform_norm is not None:
            tensors.append(self.low_platform_norm)
        return torch.cat(tensors, dim=1)


@dataclass
class DirectRegressionPredictions:
    normalized_params: torch.Tensor

    def as_tensor(self) -> torch.Tensor:
        return self.normalized_params


class ConvPatchTokenizer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stem_channels: int,
        hidden_channels: int,
        patch_size: int,
        patch_stride: int,
        variant: str = "default",
    ):
        super().__init__()
        self.variant = str(variant).lower()
        if self.variant == "simple":
            self.frontend = nn.Sequential(
                nn.Conv1d(in_channels, stem_channels, kernel_size=7, padding=3),
                nn.GELU(),
            )
        else:
            self.frontend = nn.Sequential(
                nn.Conv1d(in_channels, stem_channels, kernel_size=7, padding=3),
                nn.BatchNorm1d(stem_channels),
                nn.GELU(),
                nn.Conv1d(stem_channels, stem_channels, kernel_size=5, padding=2, groups=stem_channels),
                nn.Conv1d(stem_channels, stem_channels, kernel_size=1),
                nn.BatchNorm1d(stem_channels),
                nn.GELU(),
            )
        self.patch_embed = nn.Conv1d(
            stem_channels,
            hidden_channels,
            kernel_size=patch_size,
            stride=patch_stride,
        )
        self.norm = nn.LayerNorm(hidden_channels)

    def forward(self, iq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        local_features = self.frontend(iq)
        tokens = self.patch_embed(local_features).transpose(1, 2)
        return self.norm(tokens), local_features


class TransformerStage(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        attention_heads: int,
        num_layers: int,
        feedforward_multiplier: float,
        dropout: float,
    ):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_channels))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=attention_heads,
            dim_feedforward=int(hidden_channels * feedforward_multiplier),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_channels)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden = tokens.shape
        cls = self.cls_token.expand(batch_size, -1, -1)
        cls_pos = torch.zeros(batch_size, 1, hidden, device=tokens.device, dtype=tokens.dtype)
        token_pos = _build_sinusoidal_encoding(seq_len, hidden, tokens.device, tokens.dtype).unsqueeze(0)
        encoded = torch.cat([cls + cls_pos, tokens + token_pos], dim=1)
        encoded = self.norm(self.encoder(encoded))
        return encoded[:, 0], encoded[:, 1:]


class FeatureConditioner(nn.Module):
    def __init__(self, embedding_dim: int, hidden_channels: int):
        super().__init__()
        self.to_scale = nn.Linear(embedding_dim, hidden_channels)
        self.to_bias = nn.Linear(embedding_dim, hidden_channels)

    def forward(self, tokens: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        scale = 0.35 * torch.tanh(self.to_scale(embedding)).unsqueeze(1)
        bias = 0.08 * self.to_bias(embedding).unsqueeze(1)
        return tokens * (1.0 + scale) + bias


class TokenAttentionPooling(nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.score = nn.Linear(hidden_channels, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(tokens), dim=1)
        return torch.sum(tokens * weights, dim=1)


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.activation(out + identity)


class TemporalBlock1D(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        return self.activation(out + identity)


class DenseLayer1D(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int, bn_size: int = 4, dropout: float = 0.0):
        super().__init__()
        inter_channels = bn_size * growth_rate
        self.norm1 = nn.BatchNorm1d(in_channels)
        self.act1 = nn.GELU()
        self.conv1 = nn.Conv1d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm1d(inter_channels)
        self.act2 = nn.GELU()
        self.conv2 = nn.Conv1d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.conv1(self.act1(self.norm1(x)))
        new_features = self.conv2(self.act2(self.norm2(new_features)))
        new_features = self.dropout(new_features)
        return torch.cat([x, new_features], dim=1)


class Transition1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool1d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class PGIQNet(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        summary_dim = self.config.hidden_channels * 4

        self.tokenizer = ConvPatchTokenizer(
            in_channels=self.config.input_channels,
            stem_channels=self.config.stem_channels,
            hidden_channels=self.config.hidden_channels,
            patch_size=self.config.patch_size,
            patch_stride=self.config.patch_stride,
            variant=self.config.tokenizer_variant,
        )
        self.shared_encoder = TransformerStage(
            hidden_channels=self.config.hidden_channels,
            attention_heads=self.config.attention_heads,
            num_layers=self.config.shared_transformer_layers,
            feedforward_multiplier=self.config.feedforward_multiplier,
            dropout=self.config.dropout,
        )

        self.ts_attn_pool = TokenAttentionPooling(self.config.hidden_channels) if self.config.ts_pooling == "attn" else None
        ts_input_dim = self._ts_summary_dim()
        self.ts_head = nn.Sequential(
            nn.Linear(ts_input_dim, self.config.hidden_channels),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_channels, self.config.ts_embedding_dim),
            nn.GELU(),
        )
        self.ts_out = nn.Sequential(
            nn.Linear(self.config.ts_embedding_dim, 1),
            nn.Sigmoid(),
        )

        tx_input_dim = summary_dim + self.config.ts_embedding_dim
        if self.config.use_local_summary:
            self.local_summary_proj = nn.Sequential(
                nn.Linear(self.config.stem_channels * 3, self.config.local_summary_dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
            )
            tx_input_dim += self.config.local_summary_dim
        else:
            self.local_summary_proj = None

        self.conditioner = FeatureConditioner(self.config.ts_embedding_dim, self.config.hidden_channels)
        self.tx_decoder = TransformerStage(
            hidden_channels=self.config.hidden_channels,
            attention_heads=self.config.attention_heads,
            num_layers=self.config.tx_transformer_layers,
            feedforward_multiplier=self.config.feedforward_multiplier,
            dropout=self.config.dropout,
        )
        self.tx_head = nn.Sequential(
            nn.Linear(tx_input_dim, self.config.hidden_channels),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_channels, self.config.hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_channels // 2, 2),
            nn.Sigmoid(),
        )

    def forward(self, iq: torch.Tensor) -> TwoStagePredictions:
        tokens, local_features = self.tokenizer(iq)
        ts_token, shared_tokens = self.shared_encoder(tokens)
        shared_summary = self._summarize(ts_token, shared_tokens)
        ts_summary = self._summarize_ts(ts_token, shared_tokens)

        ts_embedding = self.ts_head(ts_summary)
        ts_norm = self.ts_out(ts_embedding)

        conditioned_tokens = self.conditioner(shared_tokens, ts_embedding)
        tx_token, tx_tokens = self.tx_decoder(conditioned_tokens)
        tx_summary = self._summarize(tx_token, tx_tokens)
        tx_inputs = [tx_summary, ts_embedding]
        if self.local_summary_proj is not None:
            tx_inputs.append(self._summarize_local(local_features))
        duty_x = self.tx_head(torch.cat(tx_inputs, dim=1))
        return TwoStagePredictions(
            ts_norm=ts_norm,
            timing_param=duty_x[:, 0:1],
            x_norm=duty_x[:, 1:2],
            ts_embedding=ts_embedding,
        )

    def shared_modules(self) -> list[nn.Module]:
        return [self.tokenizer, self.shared_encoder]

    def ts_modules(self) -> list[nn.Module]:
        modules = [self.ts_head, self.ts_out]
        if self.ts_attn_pool is not None:
            modules.append(self.ts_attn_pool)
        return modules

    def tx_modules(self) -> list[nn.Module]:
        return [self.conditioner, self.tx_decoder, self.tx_head]

    def _summarize(self, cls_token: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        token_mean = torch.mean(tokens, dim=1)
        token_max = torch.amax(tokens, dim=1)
        token_std = torch.std(tokens, dim=1, correction=0)
        return torch.cat([cls_token, token_mean, token_max, token_std], dim=1)

    def _summarize_ts(self, cls_token: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        if self.config.ts_pooling == "cls":
            return cls_token
        if self.config.ts_pooling == "attn":
            attn_summary = self.ts_attn_pool(tokens)
            return torch.cat([cls_token, attn_summary], dim=1)
        return self._summarize(cls_token, tokens)

    def _summarize_local(self, local_features: torch.Tensor) -> torch.Tensor:
        local_mean = torch.mean(local_features, dim=2)
        local_max = torch.amax(local_features, dim=2)
        local_std = torch.std(local_features, dim=2, correction=0)
        local_summary = torch.cat([local_mean, local_max, local_std], dim=1)
        return self.local_summary_proj(local_summary)

    def _ts_summary_dim(self) -> int:
        if self.config.ts_pooling == "cls":
            return self.config.hidden_channels
        if self.config.ts_pooling == "attn":
            return self.config.hidden_channels * 2
        return self.config.hidden_channels * 4


class ResNetRegressionNet(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig(architecture="resnet_regression", input_channels=2)
        base_channels = max(64, self.config.stem_channels)
        stage2_channels = base_channels * 2
        stage3_channels = base_channels * 4
        stage4_channels = base_channels * 8

        self.stem = nn.Sequential(
            nn.Conv1d(self.config.input_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.GELU(),
        )
        self.stage1 = nn.Sequential(
            ResidualBlock1D(base_channels, base_channels),
            ResidualBlock1D(base_channels, base_channels),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock1D(base_channels, stage2_channels, stride=2),
            ResidualBlock1D(stage2_channels, stage2_channels),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock1D(stage2_channels, stage3_channels, stride=2),
            ResidualBlock1D(stage3_channels, stage3_channels),
        )
        self.stage4 = nn.Sequential(
            ResidualBlock1D(stage3_channels, stage4_channels, stride=2),
            ResidualBlock1D(stage4_channels, stage4_channels),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(stage4_channels, self.config.hidden_channels),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_channels, 3),
            nn.Sigmoid(),
        )

    def forward(self, iq: torch.Tensor) -> DirectRegressionPredictions:
        x = self.stem(iq)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).squeeze(-1)
        return DirectRegressionPredictions(normalized_params=self.head(x))


class TCNRegressionNet(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig(architecture="tcn_regression", input_channels=2)
        channels = max(self.config.hidden_channels, 128)

        self.stem = nn.Sequential(
            nn.Conv1d(self.config.input_channels, channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                TemporalBlock1D(channels, dilation=1, dropout=self.config.dropout),
                TemporalBlock1D(channels, dilation=2, dropout=self.config.dropout),
                TemporalBlock1D(channels, dilation=4, dropout=self.config.dropout),
                TemporalBlock1D(channels, dilation=8, dropout=self.config.dropout),
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(channels, self.config.hidden_channels),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_channels, 3),
            nn.Sigmoid(),
        )

    def forward(self, iq: torch.Tensor) -> DirectRegressionPredictions:
        x = self.stem(iq)
        for block in self.blocks:
            x = block(x)
        x = self.pool(x).squeeze(-1)
        return DirectRegressionPredictions(normalized_params=self.head(x))


class DenseNetRegressionNet(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig(architecture="densenet_regression", input_channels=2)
        growth_rate = 32
        block_config = [6, 12, 24, 16]
        bn_size = 4
        init_features = 64

        self.stem = nn.Sequential(
            nn.Conv1d(self.config.input_channels, init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(init_features),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        num_features = init_features
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        for idx, num_layers in enumerate(block_config):
            block = nn.Sequential(
                *[
                    DenseLayer1D(
                        in_channels=num_features + i * growth_rate,
                        growth_rate=growth_rate,
                        bn_size=bn_size,
                        dropout=self.config.dropout,
                    )
                    for i in range(num_layers)
                ]
            )
            self.blocks.append(block)
            num_features = num_features + num_layers * growth_rate
            if idx != len(block_config) - 1:
                out_features = num_features // 2
                self.transitions.append(Transition1D(num_features, out_features))
                num_features = out_features

        self.final_num_features = num_features
        self.final_norm = nn.BatchNorm1d(num_features)
        self.final_act = nn.GELU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(num_features, self.config.hidden_channels),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_channels, 3),
            nn.Sigmoid(),
        )

    def forward(self, iq: torch.Tensor) -> DirectRegressionPredictions:
        x = self.stem(iq)
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx < len(self.transitions):
                x = self.transitions[idx](x)
        x = self.final_act(self.final_norm(x))
        x = self.pool(x).squeeze(-1)
        return DirectRegressionPredictions(normalized_params=self.classifier(x))


class GateReconstructionNet(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig(architecture="gate_reconstruction", input_channels=2)
        summary_dim = self.config.hidden_channels * 4

        self.tokenizer = ConvPatchTokenizer(
            in_channels=self.config.input_channels,
            stem_channels=self.config.stem_channels,
            hidden_channels=self.config.hidden_channels,
            patch_size=self.config.patch_size,
            patch_stride=self.config.patch_stride,
            variant=self.config.tokenizer_variant,
        )
        self.shared_encoder = TransformerStage(
            hidden_channels=self.config.hidden_channels,
            attention_heads=self.config.attention_heads,
            num_layers=self.config.shared_transformer_layers,
            feedforward_multiplier=self.config.feedforward_multiplier,
            dropout=self.config.dropout,
        )
        self.ts_head = nn.Sequential(
            nn.Linear(summary_dim, self.config.hidden_channels),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_channels, self.config.ts_embedding_dim),
            nn.GELU(),
        )
        self.ts_out = nn.Sequential(
            nn.Linear(self.config.ts_embedding_dim, 1),
            nn.Sigmoid(),
        )
        gate_input_dim = summary_dim + self.config.ts_embedding_dim
        self.x_head = nn.Sequential(
            nn.Linear(gate_input_dim, self.config.hidden_channels),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_channels, 1),
            nn.Sigmoid(),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(gate_input_dim, self.config.hidden_channels),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_channels, self._gate_head_output_dim()),
            nn.Sigmoid(),
        )

    def forward(self, iq: torch.Tensor) -> GateReconstructionPredictions:
        tokens, _ = self.tokenizer(iq)
        ts_token, shared_tokens = self.shared_encoder(tokens)
        shared_summary = _summarize_tokens(ts_token, shared_tokens)
        ts_embedding = self.ts_head(shared_summary)
        gate_inputs = torch.cat([shared_summary, ts_embedding], dim=1)
        gate_outputs = self.gate_head(gate_inputs)
        if self._uses_dual_platform_gate():
            gate_period = gate_outputs[:, : self.config.gate_bins]
            low_platform_norm = gate_outputs[:, self.config.gate_bins :]
        else:
            gate_period = gate_outputs
            low_platform_norm = None
        return GateReconstructionPredictions(
            ts_norm=self.ts_out(ts_embedding),
            x_norm=self.x_head(gate_inputs),
            gate_period=gate_period,
            low_platform_norm=low_platform_norm,
        )

    def _uses_dual_platform_gate(self) -> bool:
        return str(self.config.gate_representation).lower() == "dual_platform_gate"

    def _gate_head_output_dim(self) -> int:
        if self._uses_dual_platform_gate():
            return self.config.gate_bins * 2
        return self.config.gate_bins


def _build_sinusoidal_encoding(length: int, hidden: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, hidden, 2, device=device, dtype=dtype) * (-math.log(10000.0) / max(hidden, 1))
    )
    encoding = torch.zeros(length, hidden, device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(position * div_term)
    encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
    return encoding


def _summarize_tokens(cls_token: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    token_mean = torch.mean(tokens, dim=1)
    token_max = torch.amax(tokens, dim=1)
    token_std = torch.std(tokens, dim=1, correction=0)
    return torch.cat([cls_token, token_mean, token_max, token_std], dim=1)


def build_model(config: ModelConfig | None = None) -> nn.Module:
    resolved = config or ModelConfig()
    architecture = str(resolved.architecture).lower()
    if architecture == "baseline_transformer":
        return PGIQNet(resolved)
    if architecture == "gate_reconstruction":
        return GateReconstructionNet(resolved)
    if architecture == "resnet_regression":
        return ResNetRegressionNet(resolved)
    if architecture == "tcn_regression":
        return TCNRegressionNet(resolved)
    if architecture == "densenet_regression":
        return DenseNetRegressionNet(resolved)
    raise ValueError(f"Unsupported architecture '{resolved.architecture}'.")
