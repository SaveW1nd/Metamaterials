# Gate-Reconstruction 架构重设计方案

## 1. 设计目标

本次重设计的目标不是继续堆复杂模块，而是把当前模型改造成：

$$
\text{参数主导} + \text{gate 可解释}
$$

具体要求如下：

1. 保留 `gate`，但 `gate` 必须有明确物理意义。
2. 模型主任务是直接预测物理参数，而不是靠复杂中间技巧间接解码。
3. 骨干网络尽量保留当前有效部分，避免无必要重写。
4. 去掉当前难解释、强耦合、强工程技巧驱动的模块和解码分支。
5. 最终文档和方法部分必须能被简洁解释，不再依赖大量“如果是这种模式就这样解码”的说明。

---

## 2. 现有架构的主要问题

当前 `gate_reconstruction` 版本的问题不在于完全不能工作，而在于**模块职责混乱**：

### 2.1 参数不是主输出，解释链过长

当前主链路是：

$$
IQ \rightarrow token \rightarrow transformer \rightarrow gate \rightarrow duty \rightarrow T_l
$$

其中：

- `T_s` 由 `ts_head` 输出
- `x` 由 `x_head` 输出
- `T_l` 不是直接预测，而是由 `gate_period` 平均值得到

这会导致两个问题：

1. `T_l` 的误差来源不直观
2. `gate` 同时承担中间表示、参数解码基础、重建对象、损失对象四种角色

### 2.2 解码层过于工程化

当前存在以下复杂分支：

- `x_decode_mode = head / template / template_mix`
- `low_platform_period / low_platform_full`
- `x_template / x_final`
- `plateau_loss / platform_consistency_loss`

这些逻辑让模型难以从“物理过程”角度讲清楚，更像补丁式增强。

### 2.3 `Tmin/Tmax` 归一化不够物理

当前 `ts_head` 先输出：

$$
\hat{t}_s \in [0,1]
$$

再映射成：

$$
\hat{T}_s = T_s^{\min} + \hat{t}_s (T_s^{\max} - T_s^{\min})
$$

这更像“数据集区间内坐标回归”，而不是物理参数估计。

### 2.4 gate 是黑箱输出，不是解释层

现在的 `gate_head` 是：

$$
\mathbf{u} \rightarrow \text{MLP} \rightarrow \hat{\mathbf g} \in [0,1]^{128}
$$

虽然能输出周期 gate，但这个 gate 是网络直接回归出来的，缺少强物理结构约束，因此解释性不足。

---

## 3. 新架构的总体思路

新的设计核心是：

$$
IQ \rightarrow 特征提取 \rightarrow 直接预测参数 \rightarrow 由参数解析生成 gate/mask
$$

即：

- 参数是主输出
- gate 是由参数推导出来的解释层
- mask 是由参数和 gate 共同解析出来的可视化/监督层

### 3.1 保留的部分

保留现有架构中最有价值且不影响解释性的部分：

- `ConvPatchTokenizer`
- 单个共享 `TransformerStage`
- `cls + mean + max + std` 的 summary 结构

也就是说，保留轻量特征骨干：

$$
IQ \rightarrow \text{ConvTokenizer} \rightarrow \text{Shared Transformer} \rightarrow \text{summary}
$$

### 3.2 删除或降级的部分

以下部分不再作为主架构组成：

- 当前 `gate_head` 直接生成 128 维周期 gate 的主路径
- `gate -> duty -> T_l` 主解码方式
- `x_decode_mode`
- `template_mix`
- `dual_platform_gate`
- `low_platform_period/full`
- `plateau_loss`
- `platform_consistency_loss`

### 3.3 新的输出结构

新的主输出变成三个明确的物理参数：

- `slice_width_us = T_l`
- `sampling_interval_us = T_s`
- `modulation_floor = x`

然后解析生成：

- 周期 gate：`gate_period`
- 全长 mask：`mask_full`

---

## 4. 新模型模块设计

## 4.1 输入层

输入保持不变：

$$
\mathbf{X} \in \mathbb{R}^{2 \times N}
$$

其中：

- 第 1 通道为 IQ 实部
- 第 2 通道为 IQ 虚部
- 当前配置下 `N = 4000`

## 4.2 Tokenizer

继续保留当前卷积 tokenizer：

$$
\mathbf{X} \rightarrow \mathbf{F}_{local} \rightarrow \mathbf{Z}
$$

其中：

- `local frontend` 提取局部时域特征
- `patch embedding` 把长序列切成 token 序列

输出：

$$
\mathbf{Z} \in \mathbb{R}^{M \times d}
$$

当前大致为：

$$
M \approx 250,\qquad d = 128
$$

## 4.3 Shared Transformer Encoder

保留一个共享 Transformer 编码器：

$$
\mathbf{Z} \rightarrow (\mathbf{c}, \mathbf{H})
$$

其中：

- `c` 为 `cls token` 的输出
- `H` 为所有 patch token 的输出

然后做 summary：

$$
\mathbf{s} = [\mathbf{c}, \operatorname{mean}(\mathbf{H}), \operatorname{max}(\mathbf{H}), \operatorname{std}(\mathbf{H})]
$$

得到：

$$
\mathbf{s} \in \mathbb{R}^{4d}
$$

当前即：

$$
\mathbf{s} \in \mathbb{R}^{512}
$$

## 4.4 参数头

新的参数头不再围绕 `gate` 展开，而是直接围绕物理参数展开。

推荐设计为三个头：

### (1) `Tl_head`

直接预测切片宽度：

$$
z_l = f_l(\mathbf{s})
$$

再用正值化输出：

$$
\hat{T}_l = T_l^{\min} + \operatorname{softplus}(z_l)
$$

最终做上界裁剪：

$$
\hat{T}_l \le T_l^{\max}
$$

### (2) `Gap_head`

不直接预测 `T_s`，而预测间隔量：

$$
z_\Delta = f_\Delta(\mathbf{s})
$$

$$
\hat{\Delta} = \Delta_{\min} + \operatorname{softplus}(z_\Delta)
$$

于是：

$$
\hat{T}_s = \hat{T}_l + \hat{\Delta}
$$

这样天然满足：

$$
\hat{T}_s \ge \hat{T}_l + \Delta_{\min}
$$

这比当前 `Sigmoid + Tmin/Tmax` 的方式更符合物理结构。

### (3) `X_head`

低平台系数仍然做有界预测：

$$
z_x = f_x(\mathbf{s})
$$

$$
\hat{x} = x_{\min} + \sigma(z_x)(x_{\max} - x_{\min})
$$

因为 `x` 本身就是有界量，这种做法合理。

---

## 5. gate 的新角色：物理解释层

新的 `gate` 不再由网络直接输出，而是由预测参数解析生成。

## 5.1 周期 gate 的定义

定义归一化周期相位：

$$
\phi \in [0,1)
$$

对应高散射区持续比例：

$$
r = \frac{\hat{T}_l}{\hat{T}_s}
$$

则理想 gate 为：

$$
g(\phi)=
\begin{cases}
1, & \phi < r \\
0, & \phi \ge r
\end{cases}
$$

为了可微训练，使用软边界版本：

$$
g(\phi) = \sigma\big(\alpha (r - \phi)\big)
$$

其中：

- `alpha` 是边界锐度系数
- 训练时可取固定值，如 `10` 到 `30`
- 推理时也可保留该软 gate，或者导出硬阈值版本

## 5.2 离散周期 gate

把一个周期离散成 `gate_bins=128` 个位置：

$$
\phi_k = \frac{k}{K-1}, \qquad K=128
$$

则：

$$
\hat{g}_k = \sigma\big(\alpha (r - \phi_k)\big)
$$

得到：

$$
\hat{\mathbf g} \in [0,1]^{128}
$$

此时每一个 bin 都有明确物理含义：

- 表示一个周期内某个相位位置是否处于高散射态

这就使 `gate` 成为真正可解释的周期门控曲线，而不是黑箱输出。

---

## 6. 全长 mask 的解析生成

对第 `n` 个采样点，对应时间：

$$
t_n = \frac{n}{f_s}
$$

考虑 jammer 延迟 `\tau_j` 后，相位为：

$$
\phi_n = \frac{\operatorname{mod}(\max(t_n-\tau_j,0), \hat{T}_s)}{\hat{T}_s}
$$

于是全长 mask 为：

$$
\hat{m}[n] = \hat{x} + (1-\hat{x}) g(\phi_n)
$$

写成向量形式：

$$
\hat{\mathbf m} \in \mathbb{R}^{N}
$$

因此：

- `gate_period` 是周期解释层
- `mask_full` 是整段信号解释层

两者都由参数解析而来，不再依赖额外神经网络 head。

---

## 7. 新损失函数设计

新损失函数应当遵循“参数为主、结构为辅”的原则。

## 7.1 主损失：参数回归

主损失为：

$$
\mathcal L_{param}
=
w_l \mathcal L_l
+
w_s \mathcal L_s
+
w_x \mathcal L_x
$$

其中：

- `L_l`：`Tl` 的回归损失
- `L_s`：`Ts` 的回归损失
- `L_x`：`x` 的回归损失

可采用 `SmoothL1`。

## 7.2 辅助损失：mask 一致性

由于 `mask_full` 由参数解析生成，可直接与真值 `jammer_mask` 对齐：

$$
\mathcal L_{mask}
=
\operatorname{SmoothL1}(\hat{\mathbf m}, \mathbf m)
$$

它的作用是：

- 强化参数的结构一致性
- 保证预测参数确实能生成合理的门控过程

## 7.3 可选辅助损失：周期 gate 一致性

根据真值参数也可以生成理论周期 gate：

$$
\mathbf g^\star
$$

然后监督：

$$
\mathcal L_{gate}
=
\operatorname{SmoothL1}(\hat{\mathbf g}, \mathbf g^\star)
$$

注意这里 `\hat{\mathbf g}` 不是网络单独预测，而是由预测参数生成。  
这一项本质上是在约束参数层与周期解释层一致。

## 7.4 总损失

最终损失可写为：

$$
\mathcal L
=
\mathcal L_{param}
+
\lambda_m \mathcal L_{mask}
+
\lambda_g \mathcal L_{gate}
$$

推荐原则：

- `L_param` 为主
- `L_mask` 为辅助
- `L_gate` 可选，权重较小

不再保留：

- `gate_tv_weight`
- `platform_consistency_weight`
- `plateau_loss_weight`
- `template_mix` 类损失

---

## 8. 训练与推理接口设计

## 8.1 新的 prediction 对象

建议定义统一输出对象，至少包括：

- `slice_width_us`
- `sampling_interval_us`
- `modulation_floor`
- `gate_period`
- `mask_full`

其中：

- 前三项是主输出
- 后两项是解释性附加输出

## 8.2 推理逻辑

推理时流程应为：

$$
IQ \rightarrow \mathbf{s} \rightarrow (\hat{T}_l,\hat{T}_s,\hat{x}) \rightarrow (\hat{\mathbf g}, \hat{\mathbf m})
$$

最终默认返回：

- 参数预测结果
- 可选的 `gate_period`
- 可选的 `mask_full`

## 8.3 现有代码需要同步清理

需要同步删掉或重构的内容：

- `x_decode_mode`
- `x_mix_alpha`
- `low_platform_norm`
- `low_platform_period`
- `low_platform_full`
- `duty_soft` 作为参数主解码依据
- 当前 `GateReconstructionPredictions` 的旧字段设计

---

## 9. 测试计划

## 9.1 单元测试

### 解析层正确性

给定真值参数 `(Tl, Ts, x)`：

- 能生成长度正确的 `gate_period`
- `gate_period` 的高平台占比应接近：

$$
\frac{T_l}{T_s}
$$

- `mask_full` 的低平台值应为 `x`
- `mask_full` 的高平台值应接近 `1`

### 参数约束正确性

验证以下条件恒成立：

$$
T_l > 0
$$

$$
T_s \ge T_l + \Delta_{\min}
$$

$$
x \in [x_{\min}, x_{\max}]
$$

## 9.2 训练 smoke test

在当前数据集上做短训练，检查：

- loss 是否稳定下降
- 参数误差是否可正常收敛
- `mask_full` 可视化是否与真值形状一致

## 9.3 消融对比测试

至少做以下对比：

1. 当前旧架构
2. 新架构（参数主导 + 解析 gate）
3. 可选：纯三参数回归无 gate 辅助版

对比指标：

- `slice_width_hit_rate`
- `sampling_interval_hit_rate`
- `modulation_floor_hit_rate`
- `joint_hit_rate`
- `low_jnr_joint_hit_rate`

## 9.4 解释性验证

随机抽样若干测试样本，检查：

- 预测参数是否与 `gate_period` 一致
- `gate_period` 是否确实表现为一个周期高低散射结构
- `mask_full` 是否能肉眼对应真实门控节奏

---

## 10. 预期收益

新的架构预计带来以下收益：

1. **解释性更强**  
   参数先预测，gate 再解析生成，逻辑顺序明确。

2. **模块职责清晰**  
   参数头负责预测，physics layer 负责解释，不再让 gate 兼任过多角色。

3. **训练目标更稳定**  
   去掉当前大量混合解码和分支逻辑，优化目标更单纯。

4. **论文表达更清楚**  
   可以直接叙述为：

$$
\text{shared encoder} \rightarrow \text{physical parameters} \rightarrow \text{analytic gate reconstruction}
$$

而不是复杂的黑箱 gate 解码故事。

---

## 11. 默认假设

本次方案默认以下前提成立：

- 最终任务仍为三参数估计：
  - `Tl`
  - `Ts`
  - `x`
- `gate` 必须保留，但角色应当是**物理解释层**
- 骨干网络继续采用当前轻量 `Conv + Transformer`
- 重构重点放在输出头、解码层、损失函数，不优先重写 tokenizer 和 backbone
- baseline 与评测口径暂不修改，等新架构稳定后再统一重跑
