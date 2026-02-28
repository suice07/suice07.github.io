---
layout:     post
title:      "RMSNorm vs LayerNorm"
subtitle:   "深度学习中两种归一化方法的对比"
date:       2024-09-05
author:     "Suice"
header-img: "img/post-bg-2015.jpg"
catalog:    true
tags:
    - Deep Learning
    - Normalization
    - Transformer
---

## 前言

在深度学习中，**归一化（Normalization）** 是稳定训练过程、加速收敛的关键技术。**LayerNorm** 长期以来是 Transformer 架构的标配，而近年来 **RMSNorm** 凭借更简洁的计算被 LLaMA、Gemma 等新一代大模型广泛采用。

本文将从数学原理、计算流程、性能对比三个维度，系统梳理这两种归一化方法的异同。

---

## 一、LayerNorm（层归一化）

### 1.1 核心思想

LayerNorm 对单个样本的**整层特征**进行归一化，计算该层所有神经元激活值的均值和方差，然后进行标准化。

### 1.2 数学公式

给定输入向量 $\mathbf{x} = (x_1, x_2, \ldots, x_N)$：

**Step 1**：计算均值

$$
\mu = \frac{1}{N} \sum_{i=1}^{N} x_i
$$

**Step 2**：计算方差

$$
\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2
$$

**Step 3**：标准化

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

**Step 4**：缩放与偏移（可学习参数）

$$
y_i = \gamma \cdot \hat{x}_i + \beta
$$

其中 $\gamma$（缩放因子）和 $\beta$（偏移量）是可学习参数，$\epsilon$ 是防止除零的小常数。

### 1.3 特点

- **信息完整**：同时利用了均值和方差，能够捕获输入的全局分布特征
- **通用性强**：适用于 NLP、CV 等多种任务场景
- **计算开销**：需要两次遍历数据（先算均值，再算方差）

---

## 二、RMSNorm（均方根归一化）

### 2.1 核心思想

RMSNorm 是 LayerNorm 的简化版本。它的核心观点是：**LayerNorm 中的均值平移（re-centering）并非必要**，去掉均值计算后仍能保持归一化的效果。

> 论文来源：*Root Mean Square Layer Normalization* (Zhang & Sennrich, 2019)

### 2.2 数学公式

给定输入向量 $\mathbf{x} = (x_1, x_2, \ldots, x_N)$：

**Step 1**：计算均方根（RMS）

$$
\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2}
$$

**Step 2**：归一化

$$
\hat{x}_i = \frac{x_i}{\text{RMS}(\mathbf{x}) + \epsilon}
$$

**Step 3**：缩放（可学习参数）

$$
y_i = \gamma \cdot \hat{x}_i
$$

> **注意**：RMSNorm 通常**不使用偏置项 $\beta$**，进一步减少了参数量。

### 2.3 特点

- **计算更简洁**：无需计算均值，只需一次遍历即可完成
- **参数更少**：去掉了偏置项 $\beta$
- **效果相当**：在实践中，性能与 LayerNorm 持平甚至更优

---

## 三、核心差异对比

### 3.1 计算流程对比

```
LayerNorm:
  x → 计算均值(μ) → 计算方差(σ²) → 标准化((x-μ)/σ) → γ·x̂ + β → y
       ↑ 两次遍历 ↑                     ↑ 两个参数 ↑

RMSNorm:
  x → 计算 RMS → 归一化(x/RMS) → γ·x̂ → y
       ↑ 一次遍历 ↑           ↑ 一个参数 ↑
```

### 3.2 差异总结

| 对比维度 | LayerNorm | RMSNorm |
|----------|-----------|---------|
| **统计量** | 均值 $\mu$ + 方差 $\sigma^2$ | 仅均方根 RMS |
| **均值平移** | ✅ 有（re-centering） | ❌ 无 |
| **可学习参数** | $\gamma$ 和 $\beta$ | 仅 $\gamma$ |
| **遍历次数** | 2 次 | 1 次 |
| **计算复杂度** | 略高 | 更低（约减少 10-15%） |
| **数值稳定性** | 好 | 好（避免了减均值带来的精度损失） |

### 3.3 为什么 RMSNorm 去掉均值仍然有效？

直观理解：

1. **训练动态自适应**：可学习参数 $\gamma$ 能够在训练过程中隐式补偿均值信息的缺失
2. **激活值分布**：在深层网络中，经过多层变换后的激活值分布往往已经近似零均值，均值平移的收益递减
3. **正则化效果**：去掉均值计算减少了模型复杂度，可能带来轻微的正则化效果

---

## 四、实际应用场景

### 4.1 使用 LayerNorm 的模型

- **BERT** 及其变体（RoBERTa、ALBERT 等）
- **GPT-2**
- **原始 Transformer**（Vaswani et al., 2017）
- 大多数计算机视觉 Transformer（ViT、Swin Transformer）

### 4.2 使用 RMSNorm 的模型

- **LLaMA / LLaMA 2 / LLaMA 3**（Meta）
- **Gemma**（Google）
- **Mistral**
- **Qwen**（阿里）
- 越来越多的新一代大语言模型

> **趋势**：在大语言模型（LLM）领域，RMSNorm 正在逐步取代 LayerNorm，成为主流选择。这主要是因为在数十亿参数的模型中，计算效率的微小提升也会带来显著的训练成本降低。

---

## 五、如何选择？

| 场景 | 推荐 | 理由 |
|------|------|------|
| **大规模 LLM 训练** | RMSNorm | 计算效率更高，已被主流模型验证 |
| **追求极致训练速度** | RMSNorm | 减少约 10-15% 的归一化计算量 |
| **需要严格复现经典模型** | LayerNorm | 原始论文使用 LayerNorm |
| **不确定时** | 两者都试 | 在你的任务上做 A/B 测试最可靠 |

---

## 总结

- **LayerNorm** 是经典且通用的归一化方法，利用均值和方差进行完整的标准化
- **RMSNorm** 是 LayerNorm 的简化版，去掉了均值计算和偏置参数，在保持效果的同时降低了计算开销
- 在大模型时代，RMSNorm 凭借更高的计算效率正逐渐成为新一代模型的标配
- 最终选择应结合具体任务和模型架构，通过实验验证来决定
