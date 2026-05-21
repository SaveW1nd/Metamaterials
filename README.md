# Metasurface-Inspired ISRJ Parameter Estimation

基于深度学习的间歇采样转发干扰（ISRJ）参数估计方法，针对超表面启发的干扰场景，从接收信号的 IQ 数据中直接估计干扰参数。

## 研究背景

间歇采样转发干扰（ISRJ）是一种针对线性调频（LFM）雷达的有效干扰方式。本项目提出基于 Transformer 的端到端参数估计框架，通过门控重建机制从接收信号中估计干扰的切片宽度、采样间隔和调制深度三个关键参数。

## 项目结构

```
src/
├── model.py          # 模型架构（PGIQNet, ResNet, TCN, DenseNet）
├── train.py          # 训练流程
├── config.py         # 配置定义
├── dataset.py        # 数据集加载
├── generator.py      # ISRJ 信号生成器
├── losses.py         # 损失函数
├── infer.py          # 推理脚本
├── baseline.py       # 基线方法
└── baseline_wei2018.py

configs/
├── train.yaml        # 训练配置
├── dataset.yaml      # 数据集配置
└── dataset_no_input_scale.yaml

tests/                # 单元测试
baseline/             # 传统方法（MATLAB）
matlab/               # MATLAB 信号仿真与绘图
IEEE_RadarConf/       # 论文源文件
```

## 模型架构

### GateReconstructionNet（主方法）

核心思路：不直接回归三个参数，而是通过重建干扰门控信号来隐式约束参数估计。

```
IQ 信号 (2, 4000)
    │
    ▼
ConvPatchTokenizer
  ├─ 前端卷积 (Conv1D + Depthwise-Separable Conv)
  └─ Patch 嵌入 → Token 序列 (N, 128)
    │
    ▼
TransformerStage (共享编码器)
  ├─ CLS Token + 正弦位置编码
  └─ 4 层 Transformer Encoder (Pre-Norm, GELU, 4-head attention)
    │
    ├──────────────────────────┐
    ▼                          ▼
  ts_head                   gate_head
  (采样间隔估计)            (门控波形重建)
    │                          │
    ▼                          ▼
  Sigmoid → Ts              gate_period (128-bin)
                               │
                            x_head → 调制深度
```

- **输入**：IQ 两路信号，长度 4000 采样点
- **Tokenizer**：卷积前端提取局部特征 → Patch 嵌入生成 token 序列
- **共享编码器**：4 层 Transformer，CLS token 聚合全局信息
- **Ts 分支**：从 CLS+统计摘要中估计采样间隔（归一化到 0-1）
- **Gate 分支**：重建 128-bin 的门控周期波形，同时估计调制深度
- **FeatureConditioner**：用 Ts embedding 对 token 做 FiLM 调制，让后续解码感知时序结构

### 基线模型

| 架构 | 结构 | 特点 |
|------|------|------|
| `resnet_regression` | ResNet-18 (1D)，4 stage × 2 blocks | 残差连接，逐 stage 下采样，全局平均池化后直接回归 3 参数 |
| `tcn_regression` | 4 层膨胀因果卷积 (dilation=1,2,4,8) | 指数增长感受野覆盖长程依赖，参数量小 |
| `densenet_regression` | DenseNet-121 (1D)，block=[6,12,24,16] | 密集连接实现特征复用，Transition 层压缩通道 |

三个基线均为端到端直接回归，输出 Sigmoid 归一化的 3 维向量 `[Ts, duty, x]`。

## 估计参数

- **slice_width_us**: 干扰切片宽度（0.4–4.0 μs）
- **sampling_interval_us**: 采样间隔（1.0–10.0 μs）
- **modulation_floor**: 调制深度下限（0.0–0.5）

## 快速开始

### 环境依赖

```bash
pip install torch numpy pyyaml
```

### 生成数据集

```bash
cd src
python dataset.py --config ../configs/dataset.yaml --output ../artifacts/demo_dataset
```

### 训练

```bash
cd src
python train.py --config ../configs/train.yaml
```

### 推理

```bash
cd src
python infer.py --checkpoint ../artifacts/checkpoints/best.pt --input <signal_file>
```

## 测试

```bash
pytest tests/
```

