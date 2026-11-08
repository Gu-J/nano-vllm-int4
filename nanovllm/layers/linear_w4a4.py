"""Single-GPU RHT-aware GPTQ linear, W128/A128, clipping=0.9.

Weights stay packed on the GPU. Each forward launches preparation followed by
the native signed INT4 GEMM. Ordinary allocations are captured by CUDA Graphs;
no activation cache or cross-request state is retained by the module.
"""
import torch
from torch import nn
import torch.distributed as dist

from nanovllm.kernels.rht_a4_tilelang import rht_quant_pack
from nanovllm.kernels.w4a4_grouped_kernel import w4a4_grouped_gemm


class W4A4Linear(nn.Module):
    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super().__init__()
        if dist.is_initialized() and dist.get_world_size() != 1:
            raise ValueError("W4A4Linear supports one GPU only")
        if bias or input_size not in (4096, 12288) or output_size % 8:
            raise ValueError("W4A4 requires bias-free Qwen3-8B projection shapes")
        self.in_features, self.out_features = input_size, output_size
        self.register_buffer("packed_weight", torch.empty(output_size, input_size // 2, dtype=torch.int8))
        self.register_buffer("weight_scale", torch.empty(output_size, input_size // 128, dtype=torch.bfloat16))
        self.register_buffer("signs", torch.empty(input_size, dtype=torch.int8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features or x.dtype != torch.bfloat16:
            raise ValueError("W4A4Linear requires BF16 activations with the configured input width")
        flat = x.reshape(-1, self.in_features).contiguous()
        m, k = flat.shape
        packed = torch.empty((m, k // 2), dtype=torch.int8, device=x.device)
        scale = torch.empty((m, k // 128), dtype=torch.bfloat16, device=x.device)
        rht_quant_pack(flat, self.signs, out_packed=packed, out_scale=scale)
        out = w4a4_grouped_gemm(packed, self.packed_weight, scale, self.weight_scale)
        return out.reshape(*x.shape[:-1], self.out_features)
