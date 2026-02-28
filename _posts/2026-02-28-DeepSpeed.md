---
layout:     post
title:      "DeepSpeed：大模型训练的基础设施"
subtitle:   "从显存困境到 ZeRO 优化器，算法工程师的实战指南"
date:       2026-02-28
author:     "Suice"
header-img: "img/post-bg-coffee.jpeg"
catalog:    true
tags:
    - Deep Learning
    - Distributed Training
    - DeepSpeed
    - LLM
---

## 前言

2020年，当你试图在单张 A100（80GB）上训练一个 7B 参数的模型时，你很快会发现：**光是模型参数 + 优化器状态就需要约 112GB 显存**，一张卡根本放不下。

这不是你代码写错了，而是大模型训练的物理现实。DeepSpeed 正是为解决这类问题而诞生的。

**DeepSpeed** 是微软于 2020 年开源的深度学习训练优化库，其核心贡献 **ZeRO（Zero Redundancy Optimizer）** 从根本上改变了数据并行训练的显存格局。本文将从算法工程师的学习视角，系统梳理 DeepSpeed 的核心原理与实战用法。

---

## 一、背景：大模型训练的三座大山

### 1.1 显存墙

训练一个 $\Phi$ 参数量的模型（混合精度 + Adam 优化器），单卡显存占用可以分解为：

| 组成部分 | 精度 | 显存占用 |
|----------|------|----------|
| 模型参数 | FP16 | $2\Phi$ bytes |
| 梯度 | FP16 | $2\Phi$ bytes |
| 优化器状态（Adam） | FP32 momentum + FP32 variance + FP32 参数副本 | $12\Phi$ bytes |
| **合计** | — | **$16\Phi$ bytes** |

以 7B 模型为例：$16 \times 7 \times 10^9 = 112\text{GB}$，远超单张 A100 的 80GB。

> 这意味着即使不考虑激活值，**单卡连模型都放不下**。

### 1.2 通信墙

传统分布式数据并行 (Distributed Data Parallel, DDP) 在每次反向传播后需要 **All-Reduce 全部梯度**，通信量为 $2\Phi$ bytes × 2（发送 + 接收）。当模型达到数十亿参数时，通信开销可能超过计算时间。

### 1.3 效率墙

即使显存够用，传统 DDP 中每张卡都持有完整的模型参数、梯度和优化器状态——这是巨大的**冗余**。8 张卡训练意味着 8 份相同的优化器状态，浪费了 7 份显存。

---

## 二、ZeRO 优化器：DeepSpeed 的核心贡献

ZeRO 的核心洞察：**数据并行中，每张卡都保存完整的模型状态是不必要的。我们可以把这些状态切分到不同的卡上，需要时再通信获取。**

### 2.1 传统数据并行（DDP）的问题

```
传统 DDP（N 张卡）:
  每张卡都完整持有：参数(2Φ) + 梯度(2Φ) + 优化器状态(12Φ) = 16Φ bytes
  N 张卡总显存占用 = N × 16Φ（大量冗余！）
```

### 2.2 ZeRO 三个阶段

ZeRO 的三个 Stage 逐步将冗余状态切分到各卡上：

| Stage | 切分内容 | 单卡显存 | 通信量（相对 DDP） |
|-------|----------|----------|---------------------|
| **Stage 1** | 优化器状态 | $4\Phi + \frac{12\Phi}{N}$ | 相同 |
| **Stage 2** | 优化器状态 + 梯度 | $2\Phi + \frac{14\Phi}{N}$ | 相同 |
| **Stage 3** | 优化器状态 + 梯度 + 参数 | $\frac{16\Phi}{N}$ | 约 1.5× |

其中 $N$ 为 GPU 数量。

#### Stage 1：切分优化器状态

- 每张卡只持有 $\frac{1}{N}$ 的优化器状态（Adam 的 momentum 和 variance）
- 前向和反向传播不受影响，参数更新后需要一次 All-Gather 同步参数
- **显存节省最安全，通信开销无增加**

#### Stage 2：切分优化器状态 + 梯度

- 在 Stage 1 基础上，每张卡也只保留 $\frac{1}{N}$ 的梯度
- 反向传播时使用 Reduce-Scatter（替代 All-Reduce），每张卡只收集自己负责的那部分梯度
- **进一步减少显存，通信模式改变但总量不变**

#### Stage 3：全切分（参数也切分）

- 参数也切分到各卡上，每张卡只保存 $\frac{1}{N}$ 的参数
- 前向和反向传播时，需要通过 All-Gather **临时拼回**当前层的完整参数，用完即释放
- **显存可完美线性缩放，但引入额外通信**

### 2.3 直观理解

以 8 张 A100（80GB）训练 7B 模型为例：

| 方案 | 单卡显存占用 | 能否训练？ |
|------|-------------|-----------|
| 传统 DDP | 112 GB | ❌ 放不下 |
| ZeRO Stage 1 | ~26 GB | ✅ 宽裕 |
| ZeRO Stage 2 | ~20 GB | ✅ 大量空间留给激活值 |
| ZeRO Stage 3 | ~14 GB | ✅ 可以训更大 batch 或更长 seq |

---

## 三、ZeRO-Offload 与 ZeRO-Infinity

### 3.1 ZeRO-Offload

当 GPU 数量有限时（比如只有 1-2 张卡），可以将部分状态 **offload 到 CPU 内存**：

- 优化器状态放在 CPU 内存（通常有 256GB+ 的 RAM）
- 参数更新在 CPU 上完成，更新后的参数传回 GPU
- **代价**：PCIe 带宽成为瓶颈，训练速度下降

### 3.2 ZeRO-Infinity

进一步将状态 offload 到 **NVMe SSD**：

- 利用 NVMe 的 TB 级存储，理论上可以训练任意大的模型
- 通过异步预取 + 流水线化隐藏 IO 延迟

| 方案 | 存储介质 | 容量 | 带宽 | 适用场景 |
|------|----------|------|------|----------|
| ZeRO Stage 3 | GPU HBM | ~80 GB/卡 | ~2 TB/s | 多卡集群 |
| ZeRO-Offload | CPU RAM | ~256 GB+ | ~25 GB/s (PCIe) | 少卡/单卡 |
| ZeRO-Infinity | NVMe SSD | ~TB 级 | ~5 GB/s | 极大模型/单节点 |

---

## 四、实战：DeepSpeed 接入指南

### 4.1 从 PyTorch 到 DeepSpeed（最小改动）

**原始 PyTorch 训练代码：**

```python
import torch

model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**接入 DeepSpeed 后（仅需改 3 行）：**

```python
import deepspeed

model = MyModel()

# 替换：用 deepspeed.initialize 包装 model 和 optimizer
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config="ds_config.json"
)

for batch in dataloader:
    loss = model_engine(batch)
    model_engine.backward(loss)       # 替换 loss.backward()
    model_engine.step()               # 替换 optimizer.step() + zero_grad()
```

### 4.2 配置文件 ds_config.json

DeepSpeed 的行为通过一个 JSON 配置文件控制：

**ZeRO Stage 2 + 混合精度（最常用配置）：**

```json
{
    "train_batch_size": 32,
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,

    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "initial_scale_power": 16
    },

    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "reduce_scatter": true,
        "overlap_comm": true,
        "contiguous_gradients": true
    },

    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    }
}
```

**关键字段说明：**

| 字段 | 说明 |
|------|------|
| `train_batch_size` | 全局总 batch size（= micro_batch × grad_accum × GPU 数） |
| `gradient_accumulation_steps` | 梯度累积步数，等效增大 batch size |
| `fp16.enabled` | 启用 FP16 混合精度训练 |
| `zero_optimization.stage` | ZeRO 阶段（0/1/2/3） |
| `overlap_comm` | 通信与计算重叠，提升效率 |

### 4.3 ZeRO Stage 3 配置

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6
    }
}
```

### 4.4 启动训练

```bash
# 单机多卡
deepspeed --num_gpus=8 train.py --deepspeed_config ds_config.json

# 多机多节点
deepspeed --num_nodes=2 --num_gpus=8 \
    --hostfile hostfile.txt \
    train.py --deepspeed_config ds_config.json
```

### 4.5 Stage 选择建议

| 场景 | 推荐 Stage | 理由 |
|------|-----------|------|
| 模型 < 1B，卡够用 | **Stage 0 或 1** | DDP 够用，Stage 1 节省优化器显存 |
| 模型 1B–13B，多卡 | **Stage 2** | 性价比最高，通信无额外开销 |
| 模型 13B+，显存紧张 | **Stage 3** | 参数也切分，换取更大模型 |
| 单卡训大模型 | **Stage 3 + Offload** | 将状态 offload 到 CPU/NVMe |

---

## 五、混合精度训练与梯度累积

### 5.1 DeepSpeed 的混合精度支持

DeepSpeed 内置了高效的混合精度训练方案：

```json
{
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    }
}
```

如果你的硬件支持 BF16（如 A100、H100），建议使用 BF16 以避免 loss scaling 的复杂性：

```json
{
    "bf16": {
        "enabled": true
    }
}
```

| 精度 | 优势 | 注意事项 |
|------|------|----------|
| **FP16** | 广泛支持，显存减半 | 需要 loss scaling 防止下溢 |
| **BF16** | 动态范围与 FP32 相同，无需 loss scaling | 需要 Ampere+ 架构 |

### 5.2 梯度累积

在显存不足以容纳大 batch 时，梯度累积是关键技巧。DeepSpeed 原生支持：

```json
{
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 8
}
```

等效全局 batch size = `4 × 8 × num_gpus`。梯度在 8 个 micro-batch 上累积后才进行一次参数更新，减少了通信频率。

---

## 六、DeepSpeed 在生态中的位置

### 6.1 与其他方案的对比

| 方案 | 开发方 | 核心定位 | 并行策略 | 易用性 |
|------|--------|----------|----------|--------|
| **DeepSpeed** | Microsoft | ZeRO 数据并行优化 | ZeRO Stage 1/2/3 | ⭐⭐⭐⭐⭐ 接入简单 |
| **FSDP** | Meta (PyTorch) | PyTorch 原生 ZeRO-3 | 类似 ZeRO Stage 3 | ⭐⭐⭐⭐ 原生集成 |
| **Megatron-LM** | NVIDIA | 模型并行（张量/流水线） | TP + PP | ⭐⭐ 专业级 |
| **Colossal-AI** | HPC-AI Tech | 统一并行方案 | TP + PP + ZeRO | ⭐⭐⭐ |

### 6.2 如何选择？

- **纯数据并行场景**（模型能放进单卡或通过 ZeRO 切分）→ **DeepSpeed** 或 **FSDP**
- **模型必须做张量并行**（单层就超出单卡显存）→ **Megatron-LM** 或 Megatron-DeepSpeed
- **生态绑定 PyTorch**，不想引入外部依赖 → **FSDP**
- **HuggingFace 用户** → DeepSpeed 有一流集成（`Trainer` 直接传 `deepspeed` 参数）

### 6.3 与 HuggingFace 的集成

如果你使用 HuggingFace Transformers，接入 DeepSpeed 极其简单：

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    deepspeed="ds_config.json",    # 只需加这一行
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
```

---

## 七、延伸：DeepSpeed 的其他能力

DeepSpeed 不仅仅是 ZeRO 优化器，它还提供了一系列辅助功能：

| 功能 | 说明 |
|------|------|
| **DeepSpeed-Chat** | 基于 RLHF 的完整训练 pipeline（SFT → Reward → PPO） |
| **DeepSpeed-Inference** | 推理优化，支持张量并行和自定义 kernel |
| **Sparse Attention** | 稀疏注意力实现，降低长序列计算成本 |
| **1-bit Adam / LAMB** | 压缩通信的优化器，减少分布式训练的通信量 |
| **Activation Checkpointing** | 用计算换显存，减少激活值的存储开销 |

---

## 总结

DeepSpeed 的核心价值可以用一句话概括：

> **让算法工程师不需要成为分布式系统专家，就能训练超出单卡显存限制的大模型。**

从工程实践角度：

- **ZeRO Stage 2** 是最常用的配置，性价比最高，几乎无额外通信开销
- **ZeRO Stage 3 + Offload** 是「穷人的大模型训练方案」，让有限硬件也能跑大模型
- 接入成本极低：**3 行代码 + 1 个 JSON 配置文件**

从学习角度，DeepSpeed 揭示了一个重要的工程思维：**大模型训练的核心瓶颈不是算力，而是显存。** 理解「参数、梯度、优化器状态」三者的显存占比，是选择正确训练策略的基础。无论你用 DeepSpeed、FSDP 还是其他框架，这个分析框架都是通用的。
