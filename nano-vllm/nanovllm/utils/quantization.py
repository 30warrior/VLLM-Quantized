import torch
import torch.nn as nn

try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False


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
    Params4bit/Int8Params 在 F.linear 中自动触发反量化计算。
    (省显存但不提速)
    """
    if not BITSANDBYTES_AVAILABLE:
        raise ImportError("bitsandbytes is required for quantization")

    for name, param in list(model.named_parameters(recurse=True)):
        if "weight" not in name or param.ndim < 2:
            continue

        parts = name.split(".")
        attr = parts[-1]
        parent_parts = parts[:-1]
        parent_mod = model
        for p in parent_parts:
            parent_mod = getattr(parent_mod, p)

        if quant_config.load_in_4bit:
            quantized = bnb.nn.Params4bit(
                param.data,
                quant_type=quant_config.bnb_4bit_quant_type,
            )
        elif quant_config.load_in_8bit:
            quantized = bnb.nn.Int8Params(param.data)
        else:
            continue

        setattr(parent_mod, attr, quantized)


def convert_model_to_int8(model: nn.Module, quant_config: QuantizationConfig):
    """
    将模型中所有自定义线性层替换为 bnb.nn.Linear8bitLt。
    使用 INT8 Tensor Core 做矩阵乘法，同时省显存 + 提速。
    仅支持 tensor_parallel_size=1。
    """
    if not BITSANDBYTES_AVAILABLE:
        raise ImportError("bitsandbytes is required for INT8 conversion")

    _replace_linear_layers(model, quant_config)


def _replace_linear_layers(module: nn.Module, quant_config: QuantizationConfig):
    """递归替换子模块中的自定义线性层为 Linear8bitLt"""
    from nanovllm.layers.linear import LinearBase

    for name, child in list(module.named_children()):
        if isinstance(child, LinearBase):
            in_features = child.weight.shape[1]
            out_features = child.weight.shape[0]
            has_bias = child.bias is not None

            int8_linear = bnb.nn.Linear8bitLt(
                in_features, out_features,
                bias=has_bias,
                threshold=quant_config.llm_int8_threshold,
                has_fp16_weights=False,
            )
            int8_linear = int8_linear.to(child.weight.device, child.weight.dtype)
            int8_linear.weight = bnb.nn.Int8Params(child.weight.data, requires_grad=False)
            if has_bias:
                int8_linear.bias = nn.Parameter(child.bias.data)

            setattr(module, name, int8_linear)
        else:
            _replace_linear_layers(child, quant_config)


def convert_model_to_int4(model: nn.Module, quant_config: QuantizationConfig):
    """
    将模型中所有自定义线性层替换为 bnb.nn.Linear4bit。
    真正节省 75% 显存（NF4 4-bit 存储，不保留 FP16 副本）。
    仅支持 tensor_parallel_size=1。
    """
    if not BITSANDBYTES_AVAILABLE:
        raise ImportError("bitsandbytes is required for INT4 conversion")

    from nanovllm.layers.linear import LinearBase

    for name, child in list(model.named_children()):
        if isinstance(child, LinearBase):
            in_features = child.weight.shape[1]
            out_features = child.weight.shape[0]
            has_bias = child.bias is not None

            # 先将原始权重拷到 CPU，释放 GPU 上的 FP16 存储
            weight_cpu = child.weight.data.cpu()
            bias_cpu = child.bias.data.cpu() if has_bias else None
            del child
            torch.cuda.empty_cache()

            int4_linear = bnb.nn.Linear4bit(
                in_features, out_features,
                bias=False,  # 先不设 bias，后续手动加
                quant_type=quant_config.bnb_4bit_quant_type,
                compute_dtype=quant_config.bnb_4bit_compute_dtype,
            )
            int4_linear = int4_linear.to("cuda")
            # 用真实权重量化替换随机初始化的权重
            int4_linear.weight = bnb.nn.Params4bit(
                weight_cpu.to("cuda"),
                quant_type=quant_config.bnb_4bit_quant_type,
                requires_grad=False,
            )
            if has_bias:
                int4_linear.bias = nn.Parameter(bias_cpu.to("cuda"))
            else:
                int4_linear.bias = None

            setattr(model, name, int4_linear)
        else:
            convert_model_to_int4(child, quant_config)


def convert_model_to_custom_int4(model: nn.Module):
    """将模型中所有 LinearBase 子类的 FP16 权重替换为离线 INT4 量化版本。"""
    from nanovllm.layers.linear import LinearBase

    for _, child in list(model.named_modules()):
        if isinstance(child, LinearBase) and not child.use_int4:
            child.quantize_weights()


def convert_model_to_custom_int8(model: nn.Module):
    """将模型中所有 LinearBase 子类的 FP16 权重替换为离线 INT8 量化版本。"""
    from nanovllm.layers.linear import LinearBase

    for _, child in list(model.named_modules()):
        if isinstance(child, LinearBase) and not child.use_int8:
            child.quantize_weights_int8()


def print_memory_report(model: nn.Module, quant_type: str | None):
    """
    打印模型参数内存占用报告。
    显示参数总数、全精度(F16)占用、量化后占用和节省量。
    """
    num_params = 0
    num_weight_params = 0
    for param in model.parameters():
        num_params += param.numel()
    for name, param in model.named_parameters():
        # 常规 weight 参数 或 量化的 int 类型参数都计入权重参数量
        if param.ndim >= 2 and (("weight" in name) or param.dtype in (torch.int8, torch.uint8)):
            multiplier = 2  # 当作 fp16 等效参数量
            if param.dtype == torch.uint8:
                multiplier = 4  # 每个 uint8 存 2 个 INT4 → 4 个 fp16 元素
            elif param.dtype == torch.int8:
                multiplier = 2  # 每个 int8 存 1 个 INT8 → 2 个 fp16 元素
            num_weight_params += param.numel() * multiplier // 2

    fp16_gb = num_weight_params * 2 / (1024**3)
    actual_gb = fp16_gb  # 默认同 FP16

    label = "无量化 (FP16)"
    if quant_type == "int4":
        actual_gb = num_weight_params * 0.5 / (1024**3)
        label = "INT4"
    elif quant_type == "int8":
        actual_gb = num_weight_params * 1 / (1024**3)
        label = "INT8"

    saved_gb = fp16_gb - actual_gb
    saved_pct = (1 - actual_gb / fp16_gb) * 100 if fp16_gb > 0 else 0

    print(f"[显存报告] 权重参数: {num_weight_params/1e6:.0f}M  | "
          f"总参数: {num_params/1e6:.0f}M")
    print(f"            {label}占用: {actual_gb:.2f} GB  "
          f"(原FP16: {fp16_gb:.2f} GB)")
    print(f"            节省: {saved_gb:.2f} GB  ({saved_pct:.0f}%)  "
          f"+ KV Cache 额外占用")
    print(f"            GPU 总计: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
