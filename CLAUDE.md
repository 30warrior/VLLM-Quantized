# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nano-vLLM is a lightweight vLLM implementation built from scratch, providing fast offline inference with comparable speeds to vLLM. The project is designed to be readable and educational, with approximately 1,200 lines of clean Python code implementing key optimizations like prefix caching, tensor parallelism, torch compilation, and CUDA graphs.

## Installation and Setup

```bash
# Install the package from the repository
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git

# Or install from local source
pip install -e .

# Download model weights (example for Qwen3-0.6B)
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Key Commands

### Basic Usage
```python
from nanovllm import LLM, SamplingParams

# Initialize the LLM
llm = LLM("/path/to/model", enforce_eager=True, tensor_parallel_size=1)

# Set sampling parameters
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)

# Generate text
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
print(outputs[0]["text"])
```

### Running Examples and Benchmarks
```bash
# Run the example
python example.py

# Run benchmark
python bench.py

# Run with custom parameters
python bench.py  # Uses default Qwen3-0.6B model
```

### Development Commands
```bash
# Install in development mode
pip install -e .

# Run tests (if available)
python -m pytest tests/

# Check code formatting
black nanovllm/
isort nanovllm/

# Type checking
mypy nanovllm/
```

## Architecture Overview

### Core Components

1. **LLM Engine** (`nanovllm/engine/llm_engine.py`)
   - Main entry point for inference
   - Manages request scheduling and generation loop
   - Handles tokenizer and multiprocessing coordination

2. **Model Runner** (`nanovllm/engine/model_runner.py`)
   - Handles model execution on GPU
   - Manages KV cache allocation and CUDA graph capture
   - Coordinates tensor parallelism across processes
   - Implements prefill and decode batching strategies

3. **Scheduler** (`nanovllm/engine/scheduler.py`)
   - Manages sequence scheduling for optimal throughput
   - Implements chunked prefill and preemptive scheduling
   - Coordinates with block manager for memory allocation

4. **Block Manager** (`nanovllm/engine/block_manager.py`)
   - Implements prefix caching through block-level KV cache management
   - Uses xxhash for efficient block deduplication
   - Manages GPU memory allocation and deallocation

5. **Sequence Management** (`nanovllm/engine/sequence.py`)
   - Represents individual generation requests
   - Tracks token sequences, completion status, and block assignments

### Model Architecture

- **Qwen3 Implementation** (`nanovllm/models/qwen3.py`)
  - Transformer-based causal language model
  - Supports tensor parallelism for multi-GPU inference

- **Layer Implementations** (`nanovllm/layers/`)
  - Attention with flash attention support
  - Rotary position embeddings
  - Layer normalization and linear layers
  - Activation functions and sampling

### Key Optimizations

1. **Prefix Caching**: Reuses KV cache blocks for common prompt prefixes
2. **CUDA Graphs**: Captures and replays computation graphs for decode phase
3. **Tensor Parallelism**: Distributes model across multiple GPUs
4. **Chunked Prefill**: Processes long prompts in chunks to optimize memory usage
5. **Continuous Batching**: Efficiently batches multiple sequences

## Configuration

The `Config` class (`nanovllm/config.py`) manages all runtime settings:

```python
# Key configuration parameters
max_num_batched_tokens=16384    # Maximum tokens per batch
max_num_seqs=512                 # Maximum concurrent sequences
max_model_len=4096               # Maximum sequence length
gpu_memory_utilization=0.9       # GPU memory usage target
tensor_parallel_size=1           # Number of GPUs for tensor parallelism
enforce_eager=False              # Whether to disable CUDA graphs
kvcache_block_size=256           # KV cache block size
```

## File Structure

```
nanovllm/
├── __init__.py              # Package exports (LLM, SamplingParams)
├── llm.py                   # LLM class (inherits from LLMEngine)
├── config.py                # Configuration management
├── sampling_params.py       # Sampling parameter definitions
├── engine/
│   ├── llm_engine.py        # Main engine implementation
│   ├── model_runner.py      # GPU model execution
│   ├── scheduler.py         # Request scheduling
│   ├── sequence.py          # Sequence data structures
│   └── block_manager.py     # KV cache block management
├── layers/                  # Neural network layers
│   ├── attention.py
│   ├── embed_head.py
│   ├── layernorm.py
│   ├── linear.py
│   ├── rotary_embedding.py
│   ├── activation.py
│   └── sampler.py
├── models/
│   └── qwen3.py            # Qwen3 model implementation
└── utils/
    ├── context.py          # CUDA context management
    └── loader.py           # Model weight loading
```

## Dependencies

Core dependencies as defined in `pyproject.toml`:
- `torch>=2.4.0`: PyTorch for tensor operations
- `triton>=3.0.0`: GPU kernel compilation
- `transformers>=4.51.0`: Tokenizer and model utilities
- `flash-attn`: Optimized attention implementation
- `xxhash`: Fast hashing for prefix caching

## Performance Characteristics

Based on benchmark results:
- **Hardware**: RTX 4070 Laptop (8GB)
- **Model**: Qwen3-0.6B
- **Throughput**: ~1434 tokens/s (Nano-vLLM) vs ~1362 tokens/s (vLLM)
- **Memory**: Efficient KV cache management with configurable block sizes
- **Scalability**: Supports tensor parallelism for multi-GPU deployment

## Development Patterns

1. **Multi-processing**: Uses spawn context for tensor parallelism
2. **Shared Memory**: Implements efficient IPC for multi-GPU communication
3. **CUDA Optimization**: Leverages CUDA graphs for decode phase optimization
4. **Memory Management**: Implements sophisticated block-based KV cache management
5. **Batch Processing**: Optimizes throughput through continuous batching

This architecture enables Nano-vLLM to achieve performance comparable to vLLM while maintaining a clean, educational codebase that's easy to understand and modify.