from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

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
class ParameterMaskPredictions:
    slice_width_us: torch.Tensor
    sampling_interval_us: torch.Tensor
    modulation_floor: torch.Tensor

    def as_tensor(self) -> torch.Tensor:
        return torch.cat(
            [
                self.slice_width_us,
                self.sampling_interval_us,
                self.modulation_floor,
            ],
            dim=1,
        )


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
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.use_cls_token = use_cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_channels)) if use_cls_token else None
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

    def forward(self, tokens: torch.Tensor) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        _, seq_len, hidden = tokens.shape
        token_pos = _build_sinusoidal_encoding(seq_len, hidden, tokens.device, tokens.dtype).unsqueeze(0)
        encoded_inputs = tokens + token_pos
        if self.use_cls_token:
            batch_size = tokens.shape[0]
            cls = self.cls_token.expand(batch_size, -1, -1)
            cls_pos = torch.zeros(batch_size, 1, hidden, device=tokens.device, dtype=tokens.dtype)
            encoded_inputs = torch.cat([cls + cls_pos, encoded_inputs], dim=1)
            encoded = self.norm(self.encoder(encoded_inputs))
            return encoded[:, 0], encoded[:, 1:]
        encoded = self.norm(self.encoder(encoded_inputs))
        return None, encoded


class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, hidden_channels: int, attention_heads: int, dropout: float):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, hidden_channels))
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_channels)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size = tokens.shape[0]
        query = self.query.expand(batch_size, -1, -1)
        pooled, _ = self.attn(query=query, key=tokens, value=tokens, need_weights=False)
        return self.norm(pooled.squeeze(1))


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
            use_cls_token=False,
        )
        self.period_pool = MultiHeadAttentionPooling(
            hidden_channels=self.config.hidden_channels,
            attention_heads=self.config.attention_heads,
            dropout=self.config.dropout,
        )
        self.width_pool = MultiHeadAttentionPooling(
            hidden_channels=self.config.hidden_channels,
            attention_heads=self.config.attention_heads,
            dropout=self.config.dropout,
        )
        self.floor_pool = MultiHeadAttentionPooling(
            hidden_channels=self.config.hidden_channels,
            attention_heads=self.config.attention_heads,
            dropout=self.config.dropout,
        )
        self.period_head = nn.Sequential(
            nn.Linear(self.config.hidden_channels, self.config.hidden_channels),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_channels, 1),
        )
        self.tl_head = nn.Sequential(
            nn.Linear(self.config.hidden_channels * 2, self.config.hidden_channels),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_channels, 1),
        )
        self.x_head = nn.Sequential(
            nn.Linear(self.config.hidden_channels * 2, self.config.hidden_channels),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_channels, 1),
        )

    def forward(self, iq: torch.Tensor) -> ParameterMaskPredictions:
        tokens, _ = self.tokenizer(iq)
        _, shared_tokens = self.shared_encoder(tokens)

        period_context = self.period_pool(shared_tokens)
        width_context = self.width_pool(shared_tokens)
        floor_context = self.floor_pool(shared_tokens)

        period_features = period_context
        width_features = torch.cat([width_context, period_context], dim=1)
        floor_features = torch.cat([floor_context, period_context], dim=1)

        slice_width_us = torch.nn.functional.softplus(self.tl_head(width_features))
        positive_gap = torch.nn.functional.softplus(self.period_head(period_features))
        sampling_interval_us = slice_width_us + positive_gap
        modulation_floor = torch.sigmoid(self.x_head(floor_features))
        return ParameterMaskPredictions(
            slice_width_us=slice_width_us,
            sampling_interval_us=sampling_interval_us,
            modulation_floor=modulation_floor,
        )


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
