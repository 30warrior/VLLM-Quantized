import torch
import torch.nn.functional as F


def quantize_weight_fp16_to_int4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize FP16/BF16 weight to INT4 with per-channel symmetric quantization.

    Args:
        weight: [out_features, in_features] FP16 or BF16 tensor.

    Returns:
        qweight: [out_features, in_features // 2] uint8 packed tensor.
        scales: [out_features] tensor in same dtype as `weight`.
    """
    out_features, in_features = weight.shape
    assert in_features % 2 == 0, "in_features must be even for INT4 packing"
    dtype = weight.dtype

    # Per-channel symmetric scale: max_abs / 7
    absmax = weight.abs().amax(dim=1)
    absmax = torch.clamp(absmax, min=1e-12)
    scale = absmax / 7.0

    # Quantize to [-8, 7], offset by +8 → [0, 15]
    scaled = weight / scale.view(-1, 1)
    q = torch.clamp(torch.round(scaled).to(torch.int32), -8, 7)
    q = (q.to(torch.uint8) + 8).to(torch.uint8)

    # Pack: even cols → low nibble, odd cols → high nibble
    even = q[:, 0::2]
    odd = q[:, 1::2]
    qweight = even | (odd << 4)

    return qweight, scale.to(dtype)


def dequantize_weight_int4(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Dequantize INT4 packed weight back to BF16/FP16."""
    out_features, half_in = qweight.shape
    out_dtype = dtype or scales.dtype
    # Single buffer, fill directly via slicing (avoids double-sized intermediates)
    w = torch.empty(out_features, half_in * 2, dtype=out_dtype, device=qweight.device)
    w[:, 0::2] = (qweight & 0x0F).to(out_dtype)
    w[:, 1::2] = ((qweight >> 4) & 0x0F).to(out_dtype)
    w.sub_(8).mul_(scales.to(out_dtype).view(-1, 1))
    return w


@torch.no_grad()
def int4_linear_forward(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """INT4 dequantize + linear forward.

    Dequantizes packed INT4 weight on-the-fly, then does standard matmul.
    Uses `sub_` / `mul_` in-place to minimize extra allocations.
    """
    w = dequantize_weight_int4(qweight, scales, x.dtype)
    return F.linear(x, w, bias)


@torch.no_grad()
def int4_linear_chunked(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Chunked INT4 linear for memory-limited cases.

    Dequantizes and multiplies in chunks along the output dimension
    to avoid materializing the full FP16 weight at once.
    """
    out_features = qweight.shape[0]
    parts = []
    for i in range(0, out_features, chunk_size):
        end = min(i + chunk_size, out_features)
        w = dequantize_weight_int4(qweight[i:end], scales[i:end], x.dtype)
        parts.append(F.linear(x, w))

    out = torch.cat(parts, dim=-1)
    if bias is not None:
        out = out + bias
    return out


# ─── INT8 ───────────────────────────────────────────────────────


def quantize_weight_fp16_to_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize FP16/BF16 weight to INT8 with per-channel symmetric quantization.

    Args:
        weight: [out_features, in_features] FP16 or BF16 tensor.

    Returns:
        qweight: [out_features, in_features] int8 tensor.
        scales: [out_features] tensor in same dtype as `weight`.
    """
    absmax = weight.abs().amax(dim=1, keepdim=True)
    absmax = torch.clamp(absmax, min=1e-12)
    scale = absmax / 127.0
    q = torch.clamp(torch.round(weight / scale), -127, 127).to(torch.int8)
    return q, scale.squeeze(1).to(weight.dtype)


@torch.no_grad()
def int8_linear_forward(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """INT8 dequantize + linear forward."""
    w = qweight.to(x.dtype) * scales.view(-1, 1)
    return F.linear(x, w, bias)
