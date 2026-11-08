"""Native INT4 GEMM: G=128 activations, per-channel or G=128 weight scales.

Each CTA owns an output tile and completes the entire K dimension. Group dot
products use INT32 Tensor Core accumulation, followed by FP32 scaled summation.
There is no Split-K, atomic addition, reduction kernel, or output workspace.
Clipping and quantization happen before this GEMM (activation clip ratio 0.9).
"""
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

import torch
import tilelang
import tilelang.language as T


GROUP_SIZE = 128


@dataclass(frozen=True)
class GroupedConfig:
    bm: int = 32
    bn: int = 64
    stages: int = 2
    threads: int = 128
    cache_scales: bool = False
    cache_weight_scales: bool = False


@lru_cache(None)
def compile_grouped_gemm(M, N, K, cfg=GroupedConfig(), output_fp32=False, weight_grouped=False):
    BM, BN, ST, TH = cfg.bm, cfg.bn, cfg.stages, cfg.threads
    CACHE = cfg.cache_scales
    WCACHE = cfg.cache_weight_scales
    WG = weight_grouped
    G = GROUP_SIZE
    NG = K // G
    if min(M, N, K, BM, BN, TH) < 1 or ST < 0 or K % G or N % 8 or TH % 32:
        raise ValueError("Require positive dimensions, K divisible by 128, N divisible by 8, and whole warps")
    dtype = T.float32 if output_fp32 else T.bfloat16
    weight_scale_shape = (N, NG) if WG else (N,)

    @T.prim_func
    def kernel(A: T.Tensor((M, K), T.int4),
               W: T.Tensor((N, K), T.int4),
               Sa: T.Tensor((M, NG), T.bfloat16),
               Sw: T.Tensor(weight_scale_shape, T.bfloat16),
               C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=TH) as (bx, by):
            a_s = T.alloc_shared((BM, G), T.int4)
            w_s = T.alloc_shared((BN, G), T.int4)
            scales = T.alloc_shared((BM, NG + 8), T.bfloat16)
            weight_scales = T.alloc_shared((BN, NG + 8), T.bfloat16)
            partial = T.alloc_fragment((BM, BN), T.int32)
            total = T.alloc_fragment((BM, BN), T.float32)
            T.clear(total)
            if CACHE:
                for i, g in T.Parallel(BM, NG):
                    scales[i, g] = T.if_then_else(by * BM + i < M, Sa[by * BM + i, g], 0)
            if WG and WCACHE:
                for j, g in T.Parallel(BN, NG):
                    weight_scales[j, g] = T.if_then_else(bx * BN + j < N, Sw[bx * BN + j, g], 0)
            for g in T.Pipelined(NG, num_stages=ST):
                T.copy(A[by * BM, g * G], a_s)
                T.copy(W[bx * BN, g * G], w_s)
                T.gemm(a_s, w_s, partial, transpose_B=True, clear_accum=True)
                for i, j in T.Parallel(BM, BN):
                    if WG:
                        if by * BM + i < M and bx * BN + j < N:
                            activation_scale = T.cast(scales[i, g], T.float32) if CACHE else T.cast(Sa[by * BM + i, g], T.float32)
                            weight_scale = T.cast(weight_scales[j, g], T.float32) if WCACHE else T.cast(Sw[bx * BN + j, g], T.float32)
                            # BF16 x BF16 is exact in FP32; keep this product
                            # together, then fuse dot * scale + total in FP32.
                            total[i, j] += T.cast(partial[i, j], T.float32) * (activation_scale * weight_scale)
                    else:
                        if by * BM + i < M:
                            if CACHE:
                                total[i, j] += T.cast(partial[i, j], T.float32) * T.cast(scales[i, g], T.float32)
                            else:
                                total[i, j] += T.cast(partial[i, j], T.float32) * T.cast(Sa[by * BM + i, g], T.float32)
            result = T.alloc_fragment((BM, BN), dtype)
            T.clear(result)
            for i, j in T.Parallel(BM, BN):
                if by * BM + i < M and bx * BN + j < N:
                    if WG:
                        result[i, j] = total[i, j]
                    else:
                        result[i, j] = total[i, j] * T.cast(Sw[bx * BN + j], T.float32)
            T.copy(result, C[by * BM, bx * BN])
    return tilelang.compile(kernel)


@lru_cache(None)
def best_grouped_config(M, N, K, weight_grouped=False):
    path = Path(__file__).with_name("w4a4_aw_grouped_configs.json" if weight_grouped else "w4a4_grouped_configs.json")
    if path.is_file():
        item = json.loads(path.read_text()).get(f"{M},{N},{K}")
        if item:
            return GroupedConfig(**item)
    return GroupedConfig() if M <= 32 else GroupedConfig(64, 128, 2, 128)


def w4a4_grouped_gemm(a_packed, w_packed, a_scale, w_scale, out=None, config=None):
    """Packed INT4; BF16 Sa[M,K/128], Sw[N] or Sw[N,K/128], output[M,N]."""
    if a_packed.ndim != 2 or w_packed.ndim != 2 or a_packed.shape[1] != w_packed.shape[1]:
        raise ValueError("Packed inputs must have compatible [M,K/2] and [N,K/2] shapes")
    M, K = a_packed.shape[0], a_packed.shape[1] * 2
    N = w_packed.shape[0]
    if a_packed.dtype != torch.int8 or w_packed.dtype != torch.int8:
        raise TypeError("Packed inputs must use int8 storage")
    if K % GROUP_SIZE or N % 8 or min(M, N, K) < 1:
        raise ValueError("Require positive dimensions, K divisible by 128, N divisible by 8")
    weight_grouped = w_scale.ndim == 2
    expected_w_scale = (N, K // GROUP_SIZE) if weight_grouped else (N,)
    if a_scale.shape != (M, K // GROUP_SIZE) or w_scale.shape != expected_w_scale:
        raise ValueError("Require Sa[M,K/128] and Sw[N] or Sw[N,K/128]")
    if a_scale.dtype != torch.bfloat16 or w_scale.dtype != torch.bfloat16:
        raise TypeError("Both scale tensors must be BF16")
    tensors = (a_packed, w_packed, a_scale, w_scale)
    if any(not value.is_cuda or not value.is_contiguous() or value.device != a_packed.device for value in tensors):
        raise ValueError("All inputs must be contiguous CUDA tensors on one device")
    if out is None:
        out = torch.empty((M, N), device=a_packed.device, dtype=torch.bfloat16)
    if out.shape != (M, N) or out.dtype != torch.bfloat16 or out.device != a_packed.device or not out.is_contiguous():
        raise ValueError("Output must be contiguous BF16 [M,N] on the input device")
    cfg = config or best_grouped_config(M, N, K, weight_grouped)
    compile_grouped_gemm(M, N, K, cfg, weight_grouped=weight_grouped)(a_packed, w_packed, a_scale, w_scale, out)
    return out
