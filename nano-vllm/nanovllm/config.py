import os
import torch
from dataclasses import dataclass
from transformers import AutoConfig

from nanovllm.utils.quantization import QuantizationConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384 #单次推理步骤中处理的总 token 数
    max_num_seqs: int = 512 #单个调度步骤中同时处理的最大序列（请求）数量
    max_model_len: int = 4096 #定义单个请求的最大处理长度（输入prompt + 生成内容的 token 总数）
    # 所以num_seqs * max_model_len <= max_num_batched_tokens
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False #eager模式：代码依次执行，不做graph优化
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    # 量化相关参数
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    llm_int8_threshold: float = 6.0

    # 是否使用Linear8bitLt替换整个线性层（而非仅替换权重）
    use_int8_linear: bool = False
    # 是否使用Linear4bit替换整个线性层（真正节省显存）
    use_int4_linear: bool = False
    # 是否使用自定义离线INT4量化（省显存 + 保持速度，推荐）
    use_custom_int4: bool = False
    # 是否使用自定义离线INT8量化（省显存 + 精度更好）
    use_custom_int8: bool = False

    # 量化配置对象
    quantization_config: QuantizationConfig | None = None

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert not (self.use_custom_int4 and self.use_custom_int8), "Cannot enable both custom INT4 and INT8"
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)

        # 处理量化配置
        if self.load_in_4bit or self.load_in_8bit:
            # 转换字符串dtype为实际的torch dtype
            compute_dtype = getattr(torch, self.bnb_4bit_compute_dtype) if hasattr(torch, self.bnb_4bit_compute_dtype) else torch.bfloat16

            self.quantization_config = QuantizationConfig(
                load_in_4bit=self.load_in_4bit,
                load_in_8bit=self.load_in_8bit,
                bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=compute_dtype,
                llm_int8_threshold=self.llm_int8_threshold
            )
        else:
            self.quantization_config = None
