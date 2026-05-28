import torch
import torch.nn as nn

try:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Params4bit, Int8Params
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    Params4bit = None
    Int8Params = None


class QuantizationConfig:
    """量化配置"""
    def __init__(self,
                 load_in_4bit: bool = False,
                 load_in_8bit: bool = False,
                 bnb_4bit_quant_type: str = "nf4",
                 bnb_4bit_use_double_quant: bool = True,
                 bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16,
                 llm_int8_threshold: float = 6.0):

        if load_in_4bit and load_in_8bit:
            raise ValueError("Cannot enable both 4-bit and 8-bit quantization")

        if (load_in_4bit or load_in_8bit) and not BITSANDBYTES_AVAILABLE:
            raise ImportError("bitsandbytes is required for quantization")

        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.llm_int8_threshold = llm_int8_threshold

        if load_in_4bit:
            self.quant_type = "int4"
        elif load_in_8bit:
            self.quant_type = "int8"
        else:
            self.quant_type = None


def quantize_model(model: nn.Module, quant_config: QuantizationConfig):
    """
    权重加载完成后，将模型所有权重 nn.Parameter 替换为 bitsandbytes 量化版本。
    Params4bit/Int8Params 继承自 nn.Parameter，在 F.linear 中自动反量化计算。
    """
    if not BITSANDBYTES_AVAILABLE:
        raise ImportError("bitsandbytes is required for quantization")

    for name, param in list(model.named_parameters(recurse=True)):
        if "weight" not in name or param.ndim < 2:
            continue

        *parent, attr = name.rsplit(".", 1)
        parent_mod = model
        for p in parent:
            parent_mod = getattr(parent_mod, p)

        if quant_config.load_in_4bit:
            quantized = Params4bit(
                param.data,
                quant_type=quant_config.bnb_4bit_quant_type,
                compute_dtype=quant_config.bnb_4bit_compute_dtype,
            )
        elif quant_config.load_in_8bit:
            quantized = Int8Params(param.data)
        else:
            continue

        setattr(parent_mod, attr, quantized)
