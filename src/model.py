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


class ConvPatchTokenizer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stem_channels: int,
        hidden_channels: int,
        patch_size: int,
        patch_stride: int,
    ):
        super().__init__()
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
    raise ValueError(f"Unsupported architecture '{resolved.architecture}'.")
