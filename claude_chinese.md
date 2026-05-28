# CLAUDE.md

本文档为 Claude Code (claude.ai/code) 在此代码库中工作时提供指导。

## 项目概述

Nano-vLLM 是一个从零开始构建的轻量级 vLLM 实现，提供与 vLLM 相当的快速离线推理性能。该项目旨在保持代码的可读性和教育性，大约有 1,200 行干净的 Python 代码，实现了前缀缓存、张量并行、torch 编译和 CUDA 图等关键优化。

## 安装和设置

```bash
# 从代码库安装包
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git

# 或者从本地源码安装
pip install -e .

# 下载模型权重（以 Qwen3-0.6B 为例）
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## 关键命令

### 基础用法
```python
from nanovllm import LLM, SamplingParams

# 初始化 LLM
llm = LLM("/path/to/model", enforce_eager=True, tensor_parallel_size=1)

# 设置采样参数
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)

# 生成文本
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
print(outputs[0]["text"])
```

### 运行示例和基准测试
```bash
# 运行示例
python example.py

# 运行基准测试
python bench.py

# 使用自定义参数运行
python bench.py  # 使用默认 Qwen3-0.6B 模型
```

### 开发命令
```bash
# 开发模式安装
pip install -e .

# 运行测试（如果可用）
python -m pytest tests/

# 检查代码格式
black nanovllm/
isort nanovllm/

# 类型检查
mypy nanovllm/
```

## 架构概览

### 核心组件

1. **LLM 引擎** (`nanovllm/engine/llm_engine.py`)
   - 推理的主要入口点
   - 管理请求调度和生成循环
   - 处理分词器和多进程协调

2. **模型运行器** (`nanovllm/engine/model_runner.py`)
   - 在 GPU 上处理模型执行
   - 管理 KV 缓存分配和 CUDA 图捕获
   - 协调跨进程的张量并行
   - 实现预填充和解码批处理策略

3. **调度器** (`nanovllm/engine/scheduler.py`)
   - 管理序列调度以实现最佳吞吐量
   - 实现分块预填充和抢占式调度
   - 与块管理器协调内存分配

4. **块管理器** (`nanovllm/engine/block_manager.py`)
   - 通过块级 KV 缓存管理实现前缀缓存
   - 使用 xxhash 进行高效的块去重
   - 管理 GPU 内存分配和释放

5. **序列管理** (`nanovllm/engine/sequence.py`)
   - 表示单个生成请求
   - 跟踪令牌序列、完成状态和块分配

### 模型架构

- **Qwen3 实现** (`nanovllm/models/qwen3.py`)
  - 基于 Transformer 的因果语言模型
  - 支持多 GPU 推理的张量并行

- **层实现** (`nanovllm/layers/`)
  - 支持 flash attention 的注意力机制
  - 旋转位置嵌入
  - 层归一化和线性层
  - 激活函数和采样

### 关键优化

1. **前缀缓存**：为常见提示前缀重用 KV 缓存块
2. **CUDA 图**：捕获和重放解码阶段的计算图
3. **张量并行**：在多个 GPU 上分发模型
4. **分块预填充**：分块处理长提示以优化内存使用
5. **连续批处理**：高效批处理多个序列

## 配置

`Config` 类 (`nanovllm/config.py`) 管理所有运行时设置：

```python
# 关键配置参数
max_num_batched_tokens=16384    # 每批最大令牌数
max_num_seqs=512                 # 最大并发序列数
max_model_len=4096               # 最大序列长度
gpu_memory_utilization=0.9       # GPU 内存使用目标
tensor_parallel_size=1           # 用于张量并行的 GPU 数量
enforce_eager=False              # 是否禁用 CUDA 图
kvcache_block_size=256           # KV 缓存块大小
```

## 文件结构

```
nanovllm/
├── __init__.py              # 包导出 (LLM, SamplingParams)
├── llm.py                   # LLM 类（继承自 LLMEngine）
├── config.py                # 配置管理
├── sampling_params.py       # 采样参数定义
├── engine/
│   ├── llm_engine.py        # 主引擎实现
│   ├── model_runner.py      # GPU 模型执行
│   ├── scheduler.py         # 请求调度
│   ├── sequence.py          # 序列数据结构
│   └── block_manager.py     # KV 缓存块管理
├── layers/                  # 神经网络层
│   ├── attention.py
│   ├── embed_head.py
│   ├── layernorm.py
│   ├── linear.py
│   ├── rotary_embedding.py
│   ├── activation.py
│   └── sampler.py
├── models/
│   └── qwen3.py            # Qwen3 模型实现
└── utils/
    ├── context.py          # CUDA 上下文管理
    └── loader.py           # 模型权重加载
```

## 依赖项

`pyproject.toml` 中定义的核心依赖：
- `torch>=2.4.0`: 用于张量操作的 PyTorch
- `triton>=3.0.0`: GPU 内核编译
- `transformers>=4.51.0`: 分词器和模型工具
- `flash-attn`: 优化的注意力实现
- `xxhash`: 用于前缀缓存的快速哈希

## 性能特征

基于基准测试结果：
- **硬件**: RTX 4070 笔记本 (8GB)
- **模型**: Qwen3-0.6B
- **吞吐量**: ~1434 tokens/s (Nano-vLLM) vs ~1362 tokens/s (vLLM)
- **内存**: 使用可配置的块大小进行高效的 KV 缓存管理
- **可扩展性**: 支持张量并行用于多 GPU 部署

## 开发模式

1. **多进程**: 使用 spawn 上下文进行张量并行
2. **共享内存**: 为多 GPU 通信实现高效的进程间通信
3. **CUDA 优化**: 利用 CUDA 图优化解码阶段
4. **内存管理**: 实现复杂的基于块的 KV 缓存管理
5. **批处理**: 通过连续批处理优化吞吐量

这种架构使 Nano-vLLM 能够在保持清晰、易于理解和修改的教育性代码库的同时，实现与 vLLM 相当的性能。