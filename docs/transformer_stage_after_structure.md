# 当前模型中 `TransformerStage` 后的结构说明

当前实际使用的模型配置在 [A:/gfkd/Metamaterials/configs/train.yaml](/A:/gfkd/Metamaterials/configs/train.yaml)：

- `architecture: gate_reconstruction`
- `input_channels: 2`
- `hidden_channels: 128`
- `attention_heads: 4`
- `shared_transformer_layers: 4`
- `patch_size: 16`
- `patch_stride: 16`
- `ts_embedding_dim: 48`
- `gate_bins: 128`
- `gate_representation: single_gate`

对应模型实现位于 [A:/gfkd/Metamaterials/src/model.py](/A:/gfkd/Metamaterials/src/model.py#L263) 的 `GateReconstructionNet`。

## 1. `TransformerStage` 的输出

在当前配置下，输入信号长度是 `4000`，`patch_size=16`，`patch_stride=16`，所以 patch token 数量为：

```text
(4000 - 16) / 16 + 1 = 250
```

tokenizer 输出的 `tokens` 形状为：

```text
[B, 250, 128]
```

进入 `TransformerStage` 之前，会额外拼接一个 `cls token`，因此真正送入 encoder 的张量形状为：

```text
[B, 251, 128]
```

经过 4 层 `TransformerEncoder` 之后，在 [A:/gfkd/Metamaterials/src/model.py](/A:/gfkd/Metamaterials/src/model.py#L101) 被拆成两部分：

- `ts_token = encoded[:, 0]`，形状为 `[B, 128]`
- `shared_tokens = encoded[:, 1:]`，形状为 `[B, 250, 128]`

## 2. `_summarize_tokens` 汇聚模块

实现见 [A:/gfkd/Metamaterials/src/model.py](/A:/gfkd/Metamaterials/src/model.py#L350)。

输入：

- `cls_token`: `[B, 128]`
- `tokens`: `[B, 250, 128]`

模块内部包含 3 个按序列维度的统计操作：

- `token_mean`：对 250 个 token 求均值，输出 `[B, 128]`
- `token_max`：对 250 个 token 取最大值，输出 `[B, 128]`
- `token_std`：对 250 个 token 求标准差，输出 `[B, 128]`

然后将以下 4 个向量拼接：

- `cls_token`
- `token_mean`
- `token_max`
- `token_std`

得到：

```text
shared_summary: [B, 512]
```

因为：

```text
128 + 128 + 128 + 128 = 512
```

## 3. `ts_head` 模块

实现见 [A:/gfkd/Metamaterials/src/model.py](/A:/gfkd/Metamaterials/src/model.py#L284)。

输入：

```text
shared_summary: [B, 512]
```

内部结构：

```text
Linear(512 -> 128)
GELU
Dropout(0.1)
Linear(128 -> 48)
GELU
```

输出：

```text
ts_embedding: [B, 48]
```

## 4. `ts_out` 模块

实现见 [A:/gfkd/Metamaterials/src/model.py](/A:/gfkd/Metamaterials/src/model.py#L291)。

输入：

```text
ts_embedding: [B, 48]
```

内部结构：

```text
Linear(48 -> 1)
Sigmoid
```

输出：

```text
ts_norm: [B, 1]
```

## 5. 特征拼接模块

实现见 [A:/gfkd/Metamaterials/src/model.py](/A:/gfkd/Metamaterials/src/model.py#L316)。

将以下两个向量拼接：

- `shared_summary`: `[B, 512]`
- `ts_embedding`: `[B, 48]`

得到：

```text
gate_inputs: [B, 560]
```

因为：

```text
512 + 48 = 560
```

## 6. `x_head` 模块

实现见 [A:/gfkd/Metamaterials/src/model.py](/A:/gfkd/Metamaterials/src/model.py#L296)。

输入：

```text
gate_inputs: [B, 560]
```

内部结构：

```text
Linear(560 -> 128)
GELU
Dropout(0.1)
Linear(128 -> 1)
Sigmoid
```

输出：

```text
x_norm: [B, 1]
```

## 7. `gate_head` 模块

实现见 [A:/gfkd/Metamaterials/src/model.py](/A:/gfkd/Metamaterials/src/model.py#L303)。

输入：

```text
gate_inputs: [B, 560]
```

内部结构：

```text
Linear(560 -> 128)
GELU
Dropout(0.1)
Linear(128 -> 128)
Sigmoid
```

当前配置中：

- `gate_representation: single_gate`
- `gate_bins: 128`

所以输出为：

```text
gate_period: [B, 128]
```

当前模型不会再输出额外的 `low_platform_norm` 分支。

## 8. 最终输出

模型最终返回 `GateReconstructionPredictions`，见 [A:/gfkd/Metamaterials/src/model.py](/A:/gfkd/Metamaterials/src/model.py#L324)。

包含以下内容：

- `ts_norm`: `[B, 1]`
- `x_norm`: `[B, 1]`
- `gate_period`: `[B, 128]`
- `low_platform_norm`: `None`

## 9. 从 `TransformerStage` 之后开始的整体数据流

```text
4-layer TransformerStage
  -> ts_token [B,128]
  -> shared_tokens [B,250,128]
  -> summarize(cls + mean + max + std)
  -> shared_summary [B,512]
  -> ts_head
  -> ts_embedding [B,48]
  -> ts_out
  -> ts_norm [B,1]

shared_summary [B,512] + ts_embedding [B,48]
  -> concat
  -> gate_inputs [B,560]
  -> x_head
  -> x_norm [B,1]
  -> gate_head
  -> gate_period [B,128]
```

## 10. 一句话概括

当前模型在 `TransformerStage` 之后，先把 `cls token` 和所有 token 的统计量汇聚成一个 512 维摘要，再先预测一个 48 维 `ts_embedding` 和 1 维 `ts_norm`，然后把 `ts_embedding` 与摘要拼接成 560 维特征，同时送入 `x_head` 和 `gate_head`，分别输出 `x_norm` 与 128 维 `gate_period`。
