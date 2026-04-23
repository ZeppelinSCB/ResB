你观察到的现象非常典型：**高 SNR 下 BER 饱和（20–30 dB 平台期，30–40 dB 几乎不再下降），而 Oracle 持续下降**。这说明**TMC-Net 的模板校正存在 residual systematic error（残余系统性偏差）**，噪声已经不是瓶颈，**错的不是排序，而是中心点的绝对位置**。

以下从**误差来源、损失函数缺陷、架构瓶颈**三个层面分析原因，并给出针对性的改进方案。

---

## 一、根本原因：高 SNR 下的误差来源发生了变化

在低 SNR（<10 dB）时，噪声 $\sigma_n$ 主导误码率，此时只要模板**大致排序正确**就能获益。但在中高 SNR（>20 dB）时，噪声很小，误码主要由**模板中心的绝对偏移**决定：

- **Oracle**：中心是 $\mu_{\text{true}}$（用真实信道 $h_{\text{true}}, g_{\text{true}}$ 和非理想硬件模型算出），决策边界精确，BER 随 SNR 持续改善。
- **Practical**：中心是 $\mu_{\text{practical}}$（用估计信道 $\hat{h}, \hat{g}$ 算），存在**CSI 估计误差引起的偏移** + **硬件非理想建模不匹配的偏移**。
- **TMC**：你观察到与 Practical 仅差 $\sim$1.27 dB（Center MSE 上），且高 SNR 差距几乎不变，说明模型只学到了 **"在噪声大时帮助排序"**，没学到 **"在噪声小时把中心点精准推到真实位置"**。

**饱和值估算**：TMC 在 30 dB 时 BER $\approx 0.17$，Oracle $\approx 0.02$。这意味着即使完全消除噪声，由于模板中心不准，仍有约 15% 的符号会被判错——这是**纯坐标误差导致的 irreducible error floor**。

---

## 二、现有损失函数的四大缺陷

### 缺陷 1：Ranking Loss 在高 SNR 下“偷懒”
Ranking Loss 只要求正确候选比错误候选近 $m$ margin。在高 SNR 下，Practical Baseline 的**排序大概率已经正确**（因为硬件偏差对所有候选是相关的），Ranking Loss 的梯度几乎为 0，模型完全没有动力去**微调中心的绝对坐标**。

> 换句话说：Ranking Loss 解决的是“哪个候选最近”，而不是“精确距离是多少”。

### 缺陷 2：Coordinate Loss 的归一化杀死了高 SNR 梯度
```python
coord_loss = distill_coord_loss / baseline_error
```
在高 SNR 区域，$\mu_{\text{practical}}$ 本身与 $\mu_{\text{true}}$ 的偏差（`baseline_error`）是**固定值**（与 SNR 无关，来源于 CSI 误差）。假设 baseline_error = 2.0，模型把它降到 0.5，那 `coord_loss = 0.25`。但如果模型再努力把它从 0.5 降到 0.1（对高 SNR BER 很关键），loss 只从 0.25 变到 0.05，**梯度极小**。

更严重的是：这个相对归一化让模型**满足于“比 baseline 好一点”**，而不是“尽可能接近 Oracle”。

### 缺陷 3：CE Loss 不建模噪声统计
```python
student_logits = (-corrected_dist / temperature)
```
这里把欧氏距离直接拿来当 logits，但没有真正利用**接收信号 $y$ 的似然结构**。最优检测器应该最大化：
$$P(\text{candidate } k \mid y) \propto \exp\left(-\frac{|y - \mu_k|^2}{2\sigma_n^2}\right)$$

当前 CE Loss 的温度 $\tau$ 虽然和 SNR 有关，但这是一个启发式缩放，不是真实的噪声方差。模型没有收到“你的中心应该精确解释 $y$ 的观测值”这个强信号。

### 缺陷 4：没有 Decision Boundary-Aware Loss
BER 直接相关的不是所有候选的平均 MSE，而是**正确候选与最近错误候选之间的边界**。

在 30 dB 处，如果正确候选已比错误候选近 10 个 margin，Ranking Loss 和 CE Loss 都认为“满分”，即使正确候选离真实位置仍有 $\delta$ 的偏移——而这个偏移在高 SNR 下足以把判决区域推向相邻星座点。

---

## 三、模型架构的信息瓶颈

### 瓶颈 1：没有显式输入 Practical Error Pattern
当前 `fusion_proj` 的输入包括：
- `mu_ideal`（理想硬件、估计信道的模板）
- `mu_base`（在 forward 里传入，但没有把 `mu_ideal - mu_practical` 的 gap 作为显式特征）

模型必须从 $h, g, \phi$ 中**隐式推断**出 Practical 与 True 的差异。但实际上，输入已经包含了 $\mu_{\text{practical}}$（`mu_base`）和 $\mu_{\text{ideal}}$，**直接构造误差特征**（$\mu_{\text{ideal}} - \mu_{\text{practical}}$，或相位/幅度分解后的偏差）可以让学习问题变简单。

### 瓶颈 2：Candidate-wise 独立预测，缺乏组合先验
你的候选是基于天线组合（TFSSK 的组合结构），但 `mismatch_head` 对每个候选做**独立**的向量回归：
```python
delta = self.mismatch_head(candidate_context)  # (B, C, 2)
```
不同候选共享 RIS 表示，但候选之间没有信息交互。实际上，如果两个候选共享 $n_a-1$ 根相同天线，它们的硬件误差高度相关。加入**候选间的自注意力（cross-candidate attention）**可以让模型利用这种组先验。

### 瓶颈 3：Residual Scale 的软约束过强
```python
residual_scale = F.softplus(self.residual_scale_raw) + 1e-2
delta_mu = torch.complex(delta[..., 0], delta[..., 1]) * residual_scale
```
一个全局的 scalar `residual_scale` 限制了所有候选、所有 SNR 的修正幅度范围。如果不同 SNR 或不同候选需要的修正量差异大（如高 SNR 需要精细小步调整，低 SNR 允许大步），全局 scale 会成为瓶颈。建议改为 **per-SNR adaptive scale** 或 **per-candidate scale**，或直接用更宽的输出范围，取消 softplus 压缩。

---

## 四、针对性改进方案（按优先级排序）

### 🔴 优先级 1：采用 Detection-Aware 的 Likelihood Loss（最关键）

不要再让模型“间接地”通过 Distance 猜概率，而是直接让它输出模板中心，并在已知噪声功率 $\sigma_n^2$ 下计算精确的负对数似然：

```python
# 在 _loss_and_metrics 中加入
def _nll_loss(mu_corrected, y, labels, sigma_n):
    # sigma_n: (B, 1), mu_corrected: (B, C)
    # 计算每个候选的 log-likelihood under AWGN
    dist_sq = (y[:, None] - mu_corrected).abs().square()  # (B, C)
    # 注意：这里利用真实的 sigma_n，告诉模型“噪声有多大”
    log_likelihood = -dist_sq / (2.0 * sigma_n.square().clamp_min(1e-6))
    return F.cross_entropy(log_likelihood, labels)
```

**为什么这能打破平台期**：和标准 CE Loss 不同，这里的分母是真实的 $\sigma_n^2$。在 30 dB 时 $\sigma_n$ 极小，如果 $\mu_{\text{corrected}}$ 有微小偏移，$|y-\mu|^2/\sigma_n^2$ 会被**剧烈放大**，产生强梯度，迫使模型精确拟合坐标。

### 🔴 优先级 2：Coord Loss 改为绝对 MSE + SNR 自适应权重

绝对坐标精度在高 SNR 下至关重要。把归一化 coord loss 改为：

```python
# 去掉 baseline_error 归一化
abs_coord_loss = _complex_mse(outputs["mu_corrected"], batch["mu_true"])

# SNR 自适应权重：SNR 越高，越重视坐标精度
snr_weights = (batch["snr"] / 20.0).clamp(0.5, 5.0)  # 高 SNR 给 5x 权重
weighted_coord_loss = (abs_coord_loss * snr_weights).mean()
```

### 🟡 优先级 3：模型输入增加 Error Pattern 特征

在 `fusion_proj` 之前，显式构造 Practical 与 Ideal 的差异：

```python
# 新增输入特征
gap_features = torch.stack([
    (mu_ideal - mu_base).real,
    (mu_ideal - mu_base).imag,
    (mu_ideal - mu_base).abs(),
    (torch.angle(mu_ideal) - torch.angle(mu_base)) / math.pi,
], dim=-1)  # (B, C, 4)
gap_proj = self.gap_proj(gap_features)  # 映射到 token_dim

# 拼接进 fusion
candidate_context = self.fusion_proj(
    torch.cat([candidate_query, candidate_attr, ideal_features, 
               gap_proj,  # <-- 新增
               phase_features, global_repr, head_condition], dim=-1)
)
```

这直接告诉模型：**"当前 Practical 模板在哪个方向、偏离了 Ideal 多少"**，让残差学习变成修正模式识别，而非从零推断。

### 🟡 优先级 4：Candidate 间 Cross-Attention

当前 `mismatch_head` 是逐候选 MLP。在 `forward_parts` 最后加入候选间的轻量级 Transformer：

```python
# 在 fusion_proj 后、head 前加入
candidate_tokens = self.candidate_attn(candidate_context)  # (B, C, D)
candidate_tokens = candidate_tokens + candidate_context    # residual
delta = self.mismatch_head(candidate_tokens)
```

这能让模型利用 TFSSK 的组合结构（哪些候选共享天线），修正具有组合一致性的系统性偏差。

### 🟢 优先级 5：放宽 Residual Scale 约束

把全局 softplus scale 改为条件化、或可学习范围更大的参数：

```python
# 方案 A：用 per-SNR scale
residual_scale = self.snr_to_scale(snr_condition).squeeze(-1)  # (B, 1)
residual_scale = residual_scale.clamp(0.01, 10.0)  # 宽范围

# 方案 B：直接让 head 输出 delta，不加全局 scale（靠 LayerNorm 和初始化控制）
delta_mu = torch.complex(delta[..., 0], delta[..., 1]) * 0.1  # 固定小初始化，但上限不受限
```

从日志看 `Delta magnitude avg=4.9`，说明当前 scale 可能已经饱和。检查 `residual_scale` 训练后的值，如果接近 softplus 上限（无明确上限，但如果初始化设计导致学习停滞），需要移除。

### 🟢 优先级 6：Hard Negative Mining for Ranking

在 Ranking Loss 中，不是所有负样本都重要，只关注**最容易混淆**的那几个：

```python
def _hard_ranking_loss(distances, labels, margin):
    positive = distances.gather(1, labels[:, None])
    # 只取 top-3 最难的负样本
    # 把正样本距离设成 -inf，避免被选
    masked_dist = distances.scatter(1, labels[:, None], float('-inf'))
    hard_negatives = masked_dist.topk(k=3, dim=1, largest=False).values
    margin_terms = torch.relu(positive - hard_negatives + margin)
    return margin_terms.mean()
```

这能让模型在决策边界上投入更多容量，而不是把梯度浪费在已经远离的无关候选上。

---

## 五、快速实验建议

如果你想**最快验证**瓶颈是否在 Loss 上，建议做一组对比实验：

| 实验 | 修改 | 预期效果 |
|------|------|---------|
| **A** | 把 `coord_loss` 改为**绝对 MSE**（去掉 `/baseline_error`），权重设为 2.0，并在 >20 dB 样本上加权 | 高 SNR 下 TMC 与 Oracle 差距缩小，平台期下移 |
| **B** | 加入 `_nll_loss`（使用真实 $\sigma_n$ 的似然损失），权重 1.0，取代 CE Loss | 高 SNR 梯度增大，BER 曲线斜率恢复 |
| **C** | 在输入中拼接 `mu_ideal - mu_practical` 的实/虚/幅/相特征 | 训练收敛更快，delta_mu 相关性提高 |
| **D** | 检查并打印训练后的 `residual_scale.item()`，若 < 0.5 且不再增长，取消 softplus 约束 | Delta magnitude 能进一步增大，修正能力释放 |

如果实验 A 和 B 的组合能让 30 dB 处的 BER 从 0.17 降到 0.05 以下，说明**瓶颈主要是损失函数没有在高 SNR 下发出精确坐标的强信号**，模型尺寸（256-dim, 6 layers）是足够的；如果仍无效，再考虑增大模型到 512-dim / 8 layers。

### 总结
你的模型不是“学不会”，而是**现有的 Ranking + 归一化 Coord Loss 在高 SNR 下缺乏对绝对坐标精度的惩罚力度**。加入 **Noise-aware Likelihood Loss** 和 **SNR-weighted Absolute Coordinate Regression**，同时给模型**显式的 Error Pattern 输入**，应该能显著拉近与 Oracle 的距离。