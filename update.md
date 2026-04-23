# TMC-Net 高 SNR 平台期分析与解决

## 最终结论

经过系统性的消融实验，验证了以下发现：

### 1. 损失函数有效性排序

| 损失 | 当前配置 | 方案B（理想硬件） | 结论 |
|------|----------|-------------------|------|
| **Ranking Loss** | ✅ 有效 (~35%) | ✅ 有效 (~82%) | **保留** |
| **Coord Loss** | ✅ 有效 (~37%) | ✅ 有效 (~82%) | **保留** |
| CE Loss | ⚠️ 轻微有效 | ✅ 有效 | **可删除** |
| KL Divergence | ❌ **有害** (-3-5%) | - | **必须删除** |
| NLL Loss | ❌ 无效 | ✅ 有效 | **可删除** |

### 2. KL Divergence 的问题

```python
# 原 train_tmc.py 中的问题代码
distill_prob_loss = F.kl_div(
    F.log_softmax(student_logits, dim=1),      # 模型输出
    practical_baseline_probs,                    # practical_baseline（次优！准确率 ~27%）
    reduction="batchmean",
)
```

**问题**：蒸馏到次优分布会锁定模型在错误行为中。实验证明 KL Loss 会降低 3-5% 准确率。

### 3. 推荐的简化方案

已创建 `scripts/train_tmc_simple.py`，只包含有效损失：

```python
# Rank + Coord = 最优组合
rank_loss = margin_based_ranking(corrected_dist, labels, margin=0.25)
coord_loss = MSE(mu_corrected, mu_true)
loss = rank_weight * rank_loss + coord_weight * coord_loss
```

---

## 根本原因分析

### 硬件非理想误差 >> 噪声

| 指标 | 数值 | 含义 |
|------|------|------|
| `mu_practical` vs `mu_true` RMSE | **~12** | 硬件偏差 |
| 匹配 Oracle (99%) 所需 RMSE | **< 1-2** | 需要的精度 |
| 坐标误差 / 噪声比 @ 30dB | **~185x** | 坐标误差主导 |

### 信息瓶颈

模型只能观察：
- `h_hat, g_hat` - 噪声信道估计
- `phi_config` - RIS 相位配置

但 `mu_true` 依赖于：
- `h_true, g_true` - 真实信道（未知）
- 硬件非理想效应的精确建模

**这是统计上不可能完成的任务**。

---

## 方案对比

### 方案B：降低硬件非理想效应（推荐用于验证上界）

```bash
python scripts/train_tmc_simple.py \
  --ris-phase-bits 8 \
  --ris-amplitude-bias 1.0 \
  --ris-amplitude-scale 0.0 \
  --ris-coupling-decay 0.0 \
  --enable-phase-quantization False \
  --enable-amplitude-coupling False \
  --enable-mutual-coupling False \
  --csi-error-var 0.0 \
  --snr-stop 50
```

**预期效果**：
- Oracle 可达到 99%+ 准确率
- 模型可接近 Oracle 性能
- Floor 推迟到 40 dB 之后

### 方案A：轻度降低（保留研究价值）

```bash
--ris-phase-bits 5 \
--ris-amplitude-bias 0.95 \
--ris-amplitude-scale 0.02 \
--ris-coupling-decay 0.02 \
--csi-error-var 0.05
```

### 当前配置（原始硬件损伤）

```bash
python scripts/train_tmc_simple.py --epochs 80
```

**已知限制**：模型最高 ~39% 准确率，Oracle 是 99%。

---

## 架构改进的意义

即使存在信息瓶颈，架构改进仍可能有意义：

```
信息瓶颈 ≠ 无法学习
信息瓶颈 = 有性能天花板

当前状态：
- 模型准确率：~37-39%
- Oracle 上界：~99%
- 差距来源：
  1. 信息瓶颈（无法获知的真实信道）
  2. 模型欠拟合（没有充分利用可用信息）
```

**架构改进的价值**：在信息瓶颈内挖掘更多潜力。

可能的改进方向：
1. 更高效的特征提取（从 `h_hat, g_hat`）
2. 候选间跨注意力机制
3. 更好的 SNR 条件化
4. 更强的正则化防止过拟合

---

## 验证脚本

```bash
# 诊断高 SNR 平台期
python scripts/diagnose_high_snr.py --checkpoint outputs/checkpoints_simple/tmc_simple_best.pt

# 分析误差来源
python scripts/analyze_error_sources.py --checkpoint outputs/checkpoints_simple/tmc_simple_best.pt
```

---

## 消融实验数据

### 当前配置（有硬件损伤）

| 损失组合 | SNR=40dB 准确率 |
|----------|-----------------|
| Rank Only | 0.349 |
| Coord Only | 0.368 |
| **Rank + Coord** | **0.392** |
| Rank + NLL | 0.364 |
| Rank + Coord + KL | 0.332 |
| All (原版) | 0.342 |

### 方案B（理想硬件）

| 损失组合 | SNR=40dB 准确率 |
|----------|-----------------|
| Rank Only | 0.818 |
| Coord Only | 0.824 |
| **Rank + Coord** | **0.833** |
| All | 0.825 |
