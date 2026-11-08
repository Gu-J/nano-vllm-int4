"""W4A4 GEMM with per-token/per-channel BF16 scales and direct BF16 output.

Packed inputs are int8 tensors holding two signed INT4 values per byte.
Logical operation: Y[M,N] = X[M,K] @ W[N,K].T. Output is BF16.
"""

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

import torch
import tilelang
import tilelang.language as T


@dataclass(frozen=True)
class KernelConfig:
    bm: int = 32
    bn: int = 64
    bk: int = 128
    stages: int = 2
    threads: int = 128


@lru_cache(None)
def compile_gemm(M, N, K, cfg=KernelConfig(), raw_int32=False):
    """Compile one shape/configuration. raw_int32 is only a correctness oracle."""
    BM, BN, BK = cfg.bm, cfg.bn, cfg.bk
    ST, TH = cfg.stages, cfg.threads
    if min(M, N, K, BM, BN, BK, ST, TH) < 1 or K % BK or N % 8:
        raise ValueError("Require positive dimensions, N divisible by 8, K divisible by BLOCK_K")
    out_dtype = T.int32 if raw_int32 else T.bfloat16

    @T.prim_func
    def kernel(A: T.Tensor((M, K), T.int4),
               W: T.Tensor((N, K), T.int4),
               Sa: T.Tensor((M,), T.bfloat16),
               Sw: T.Tensor((N,), T.bfloat16),
               C: T.Tensor((M, N), out_dtype)):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=TH) as (bx, by):
            a_s = T.alloc_shared((BM, BK), T.int4)
            w_s = T.alloc_shared((BN, BK), T.int4)
            acc = T.alloc_fragment((BM, BN), T.int32)
            T.clear(acc)
            for ko in T.Pipelined(K // BK, num_stages=ST):
                T.copy(A[by * BM, ko * BK], a_s)
                T.copy(W[bx * BN, ko * BK], w_s)
                T.gemm(a_s, w_s, acc, transpose_B=True)
            if raw_int32:
                T.copy(acc, C[by * BM, bx * BN])
            else:
                out = T.alloc_fragment((BM, BN), T.bfloat16)
                T.clear(out)
                for i, j in T.Parallel(BM, BN):
                    if by * BM + i < M and bx * BN + j < N:
                        out[i, j] = (T.cast(acc[i, j], T.float32)
                                     * T.cast(Sa[by * BM + i], T.float32)
                                     * T.cast(Sw[bx * BN + j], T.float32))
                T.copy(out, C[by * BM, bx * BN])
    return tilelang.compile(kernel)


@lru_cache(None)
def best_config(M, N, K):
    path = Path(__file__).with_name("w4a4_configs.json")
    if path.is_file():
        entry = json.loads(path.read_text()).get(f"{M},{N},{K}")
        if entry:
            return KernelConfig(**entry)
    return KernelConfig() if M <= 32 else KernelConfig(64, 128, 128, 3, 128)


def w4a4_gemm(a_packed, w_packed, a_scale, w_scale, out=None, config=None):
    """Run GEMM and overwrite every output element.

    a_packed: [M,K/2] int8, w_packed: [N,K/2] int8.
    a_scale: [M] BF16, w_scale: [N] BF16. All tensors must be contiguous CUDA
    tensors on the same device. Output allocation is optional; callers doing
    performance measurements should supply a reusable BF16 output buffer.
    """
    if a_packed.ndim != 2 or w_packed.ndim != 2 or a_packed.shape[1] != w_packed.shape[1]:
        raise ValueError("Packed inputs must have compatible [M,K/2], [N,K/2] shapes")
    M, K = a_packed.shape[0], a_packed.shape[1] * 2
    N = w_packed.shape[0]
    if a_packed.dtype != torch.int8 or w_packed.dtype != torch.int8:
        raise TypeError("Packed weights and activations must be int8 storage")
    if a_scale.shape != (M,) or w_scale.shape != (N,):
        raise ValueError("Scale shapes must be [M] and [N]")
    if a_scale.dtype != torch.bfloat16 or w_scale.dtype != torch.bfloat16:
        raise TypeError("Both scale tensors must be BF16")
    tensors = (a_packed, w_packed, a_scale, w_scale)
    if any(not t.is_cuda or not t.is_contiguous() or t.device != a_packed.device for t in tensors):
        raise ValueError("All inputs must be contiguous CUDA tensors on one device")
    if out is None:
        out = torch.empty((M, N), device=a_packed.device, dtype=torch.bfloat16)
    if out.shape != (M, N) or out.dtype != torch.bfloat16 or out.device != a_packed.device or not out.is_contiguous():
        raise ValueError("Output must be contiguous BF16 [M,N] on the input device")
    cfg = config or best_config(M, N, K)
    kernel = compile_gemm(M, N, K, cfg)
    kernel(a_packed, w_packed, a_scale, w_scale, out)
    return out
