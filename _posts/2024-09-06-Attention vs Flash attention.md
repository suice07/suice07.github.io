---
layout:     post
title:      "Attention vs Flash Attention"
subtitle:   "从传统注意力机制到 Flash Attention 的效率革命"
date:       2024-09-06
author:     "Suice"
header-img: "img/post-bg-2015.jpg"
catalog:    true
tags:
    - Deep Learning
    - Transformer
    - Attention
---

## 前言

在 Transformer 架构中，**自注意力机制（Self-Attention）** 是核心组件，但其 $O(n^2)$ 的计算和内存复杂度在处理长序列时成为了严重瓶颈。**Flash Attention** 正是为解决这一问题而诞生的高效实现方案。

本文将从以下几个方面展开：
1. Attention 与 Self-Attention 的区别
2. 传统 Self-Attention 的计算流程与瓶颈
3. Flash Attention 的核心设计思想
4. 代码层面的对比与实现

---

## 一、Attention 与 Self-Attention

在深入 Flash Attention 之前，先厘清两个常被混淆的概念。

### 1.1 Attention（注意力机制）

Attention 是一种通用的信息聚合机制，通过计算输入序列中不同位置之间的相关性，以加权方式汇聚信息。

- **输入来源**：Query 和 Key/Value 可以来自**不同来源**（如编码器输出 → 解码器）。
- **典型场景**：序列到序列任务，如机器翻译、文本摘要。

### 1.2 Self-Attention（自注意力机制）

Self-Attention 是 Attention 的特殊形式，Query、Key、Value 均来自**同一输入序列**。

- **核心目的**：捕捉序列内部各位置之间的依赖关系。
- **典型场景**：Transformer 的编码器和解码器中的自注意力层。
- **计算复杂度**：$O(n^{2} \cdot d)$，其中 $n$ 为序列长度，$d$ 为向量维度。

> **总结**：Attention 是更广泛的概念，Self-Attention 是其在同一序列上的特例。Transformer 中的注意力计算主要指 Self-Attention。

---

## 二、传统 Self-Attention 的计算流程

### 2.1 计算步骤

传统自注意力的三步计算：

1. **计算注意力分数**：$\text{scores} = \frac{QK^T}{\sqrt{d_k}}$
2. **Softmax 归一化**：$\text{weights} = \text{softmax}(\text{scores})$
3. **加权求和**：$\text{output} = \text{weights} \cdot V$

### 2.2 代码实现

```python
import torch
import torch.nn.functional as F

def traditional_attention(Q, K, V):
    """
    Q: (batch_size, seq_len, d_k)
    K: (batch_size, seq_len, d_k)
    V: (batch_size, seq_len, d_v)
    """
    d_k = K.size(-1)
    
    # Step 1: 计算注意力分数 → (batch_size, seq_len, seq_len)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    
    # Step 2: Softmax 归一化 → (batch_size, seq_len, seq_len)
    attention_weights = F.softmax(scores, dim=-1)
    
    # Step 3: 加权求和 → (batch_size, seq_len, d_v)
    output = torch.matmul(attention_weights, V)
    
    return output
```

### 2.3 性能瓶颈

| 指标 | 复杂度 | 说明 |
|------|--------|------|
| **计算量** | $O(n^2 \cdot d)$ | 需要计算所有位置对之间的注意力分数 |
| **内存占用** | $O(n^2)$ | 注意力权重矩阵为 $n \times n$ |
| **IO 开销** | 高 | 全量矩阵需要在 HBM 和 SRAM 之间频繁搬运 |

当序列长度 $n$ 增大时（如 4K、8K 甚至更长），内存和计算开销将迅速膨胀，成为实际部署的主要瓶颈。

---

## 三、Flash Attention 的核心设计

Flash Attention 并没有改变注意力机制的数学本质，而是通过**重新编排计算顺序**和**优化内存访问模式**来实现加速。

### 3.1 核心思想：分块计算（Tiling）

将 Q、K、V 矩阵分成小块（tiles），逐块计算注意力，避免在内存中构建完整的 $n \times n$ 注意力矩阵。

**关键优化**：
- 每个小块的计算在 GPU 的**SRAM**（片上高速缓存）中完成
- 避免了将中间结果写回**HBM**（显存）的高昂 IO 开销
- 通过在线 Softmax 算法（Online Softmax）实现逐块归一化

### 3.2 内存访问优化

| 存储层级 | 容量 | 带宽 | 说明 |
|----------|------|------|------|
| **SRAM** | ~20 MB | ~19 TB/s | 片上缓存，速度极快 |
| **HBM** | ~40 GB | ~1.5 TB/s | 显存，容量大但带宽有限 |

传统 Attention 的瓶颈在于需要将 $n \times n$ 的注意力矩阵写入 HBM，而 Flash Attention 将计算限制在 SRAM 中，从而大幅减少了内存 IO。

### 3.3 在线 Softmax（Online Softmax）

传统 Softmax 需要遍历整行来计算分母（归一化常数），Flash Attention 采用了增量更新的策略：

1. 对每个新块，计算局部最大值和局部指数和
2. 与之前的结果合并，动态更新全局归一化因子
3. 最终结果与标准 Softmax 数学上完全等价

### 3.4 优势总结

| 对比维度 | 传统 Attention | Flash Attention |
|----------|---------------|-----------------|
| **内存复杂度** | $O(n^2)$ | $O(n)$ |
| **IO 复杂度** | $O(n^2)$ 次 HBM 访问 | $O(n^2 / M)$（M 为 SRAM 大小）|
| **计算结果** | 标准 | 与标准**精确一致** |
| **硬件适配** | 通用实现 | 针对 GPU 层级优化 |

---

## 四、代码对比与实现

### 4.1 传统 Attention（完整版）

```python
import torch
import torch.nn.functional as F

def traditional_attention(Q, K, V):
    """标准自注意力实现，需要 O(n^2) 内存"""
    d_k = K.size(-1)
    
    # 完整的 n×n 注意力分数矩阵
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    
    return output
```

### 4.2 Flash Attention（分块实现，PyTorch 简化版）

以下实现体现了 Flash Attention 的核心思想——分块计算与在线 Softmax：

```python
import torch
import torch.nn.functional as F

def flash_attention(Q, K, V, block_size=128):
    """
    Flash Attention 的简化实现（PyTorch 版）
    通过分块处理减少峰值内存占用

    Q: (batch_size, seq_len, d_k)
    K: (batch_size, seq_len, d_k)
    V: (batch_size, seq_len, d_v)
    block_size: 每次处理的序列块大小
    """
    batch_size, seq_len, d_k = Q.size()
    d_v = V.size(-1)
    scale = d_k ** -0.5

    # 预分配输出张量
    output = torch.zeros(batch_size, seq_len, d_v, device=Q.device)
    # 用于在线 Softmax 的辅助变量
    row_max = torch.full((batch_size, seq_len, 1), float('-inf'), device=Q.device)
    row_sum = torch.zeros(batch_size, seq_len, 1, device=Q.device)

    # 逐块遍历 K 和 V
    for j in range(0, seq_len, block_size):
        j_end = min(j + block_size, seq_len)
        K_block = K[:, j:j_end, :]  # (batch, block, d_k)
        V_block = V[:, j:j_end, :]  # (batch, block, d_v)

        # 计算当前块的注意力分数
        scores = torch.matmul(Q, K_block.transpose(-2, -1)) * scale

        # 在线 Softmax：更新最大值和指数和
        block_max = scores.max(dim=-1, keepdim=True).values
        new_max = torch.maximum(row_max, block_max)
        
        # 重新缩放之前的累积结果
        exp_old = torch.exp(row_max - new_max) * row_sum
        exp_new = torch.exp(scores - new_max)
        new_sum = exp_old + exp_new.sum(dim=-1, keepdim=True)
        
        # 更新输出
        output = output * (exp_old / new_sum) + torch.matmul(exp_new, V_block) / new_sum

        row_max = new_max
        row_sum = new_sum

    return output
```

### 4.3 使用 Flash Attention 库

在实际项目中，建议直接使用优化好的 CUDA 实现：

```python
# 安装: pip install flash-attn
from flash_attn import flash_attn_func

# Q, K, V: (batch_size, seq_len, num_heads, head_dim)
output = flash_attn_func(Q, K, V, causal=False)
```

### 4.4 关键区别一览

```
传统 Attention:
  Q ──┐
  K ──┤── 完整 n×n 矩阵（写入 HBM）── Softmax ── × V ── Output
  V ──┘

Flash Attention:
  Q ──┐
  K ──┤── 分块计算（在 SRAM 中）── 在线 Softmax ── 逐块累加 ── Output
  V ──┘
  （无需构建完整 n×n 矩阵，大幅减少 HBM 访问）
```

---

## 五、分块处理中的上下文问题

一个自然的疑问是：分块处理是否会丢失跨块的上下文信息？

### 5.1 答案：不会

Flash Attention 的分块是对**矩阵计算过程**的分块，而非对**语义序列**的分块。每个 Query 仍然会与所有 Key 计算注意力（只是分批进行），最终结果与标准 Attention **数学上完全等价**。

### 5.2 重叠分块策略（用于其他场景）

在某些**非精确近似**的注意力变体中（如 Longformer），会采用重叠窗口来保留局部上下文：

```python
def windowed_attention(sequence, block_size, overlap):
    """带重叠的窗口注意力（非 Flash Attention）"""
    outputs = []
    step = block_size - overlap
    
    for start in range(0, len(sequence), step):
        end = min(start + block_size, len(sequence))
        block = sequence[start:end]
        output = compute_attention(block)
        outputs.append(output)

    return aggregate(outputs)
```

> **注意**：这种重叠策略属于**稀疏/近似注意力**的范畴，与 Flash Attention 的精确计算是不同的概念。

---

## 总结

| 特性 | 传统 Attention | Flash Attention |
|------|---------------|-----------------|
| **数学结果** | 标准 | ✅ 完全相同 |
| **内存复杂度** | $O(n^2)$ | $O(n)$ |
| **计算效率** | 受 HBM 带宽限制 | 充分利用 SRAM |
| **实现复杂度** | 简单 | 较复杂（需 CUDA 优化） |
| **适用场景** | 短序列 | 长序列（4K+）优势明显 |

Flash Attention 的核心贡献不在于改变注意力的数学公式，而在于**重新思考了计算与存储的交互方式**。通过 IO 感知的算法设计，它让 Transformer 在处理长序列时的性能瓶颈从"计算量"转移到了真正的"计算本身"，为大模型的高效训练和推理奠定了基础。