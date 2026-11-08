"""TileLang full-width RHT + G128 A4 quantization + signed INT4 packing."""
from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path

import torch
import tilelang
import tilelang.language as T


@dataclass(frozen=True)
class Config:
    threads_per_row: int = 128
    rows_per_block: int = 1


@lru_cache(None)
def compile_kernel(K, cfg=Config()):
    TR, BR = cfg.threads_per_row, cfg.rows_per_block
    if K not in (4096, 12288) or TR not in (32, 64, 128, 256, 512) or BR < 1 or TR * BR > 1024:
        raise ValueError('Unsupported RHT configuration')
    if TR > 32 and BR * (K * 4 + K // 8) > 99 * 1024:
        raise ValueError('Shared memory exceeds SM86 per-block capacity')
    M = T.dynamic('M')
    L = K // TR
    FHT = 4096 if K == 4096 else 1024
    LOG_T, LOG_F = int(math.log2(TR)), int(math.log2(FHT))
    PER_FACTOR = 1024 // TR
    LANES = min(TR, 128)
    P = 128 // LANES
    GROUP_LOOPS = L // P
    WARP_GROUP = LANES // 32
    SHARED_K = K if TR > 32 else 1

    @T.prim_func
    def kernel(X: T.Tensor((M, K), T.bfloat16),
               D: T.Tensor((K,), T.int8),
               Packed: T.Tensor((M, K // 2), T.int8),
               Scale: T.Tensor((M, K // 128), T.bfloat16)):
        with T.Kernel(T.ceildiv(M, BR), threads=TR * BR) as block:
            thread = T.get_thread_binding()
            local_row = thread // TR
            lane = thread % TR
            row = block * BR + local_row
            a = T.alloc_local((L,), T.float32)
            b = T.alloc_local((12,), T.float32)
            shared = T.alloc_shared((BR, SHARED_K), T.float32)
            warp_max = T.alloc_shared((BR, GROUP_LOOPS * (TR // 32)), T.float32)
            low = T.alloc_local((1,), T.float32)
            high = T.alloc_local((1,), T.float32)
            temporary = T.alloc_local((1,), T.float32)
            maximum = T.alloc_local((1,), T.float32)
            scale = T.alloc_local((1,), T.float32)
            code = T.alloc_local((1,), T.int32)
            neighbor_code = T.alloc_local((1,), T.int32)
            for i in T.unroll(L):
                a[i] = T.if_then_else(row < M,
                    T.cast(X[row, i * TR + lane], T.float32) * T.cast(D[i * TR + lane], T.float32), 0.0)
            # Low five butterfly stages stay within a warp.
            for stage in T.unroll(5):
                for i in T.unroll(L):
                    temporary[0] = T.shfl_xor(a[i], 1 << stage)
                    a[i] = T.if_then_else((lane & (1 << stage)) == 0,
                                         a[i] + temporary[0], temporary[0] - a[i])
            # Only the remaining thread-index bits require block exchange.
            for stage in T.unroll(LOG_T - 5):
                for i in T.unroll(L):
                    shared[local_row, i * TR + lane] = a[i]
                T.sync_threads()
                for i in T.unroll(L):
                    temporary[0] = shared[local_row, i * TR + (lane ^ (1 << (stage + 5)))]
                    a[i] = T.if_then_else((lane & (1 << (stage + 5))) == 0,
                                         a[i] + temporary[0], temporary[0] - a[i])
                T.sync_threads()
            # High bits are register-index butterflies, with both inputs saved.
            for stage in T.unroll(LOG_F - LOG_T):
                for pair in T.unroll(L // 2):
                    index = (pair // (1 << stage)) * (2 << stage) + pair % (1 << stage)
                    low[0] = a[index]
                    high[0] = a[index + (1 << stage)]
                    a[index] = low[0] + high[0]
                    a[index + (1 << stage)] = low[0] - high[0]
            for i in T.unroll(L):
                a[i] = a[i] * T.float32(1.0 / math.sqrt(FHT))
            if K == 12288:
                # Each lane owns the 12 factors for its remaining coordinate.
                # H12[source,destination] is the frozen Paley matrix.
                for coordinate in T.unroll(PER_FACTOR):
                    for destination in T.unroll(12):
                        temporary[0] = 0.0
                        for source in T.unroll(12):
                            delta = (destination - source + 11) % 11
                            positive = (source == 0) | (destination == 0) | (delta == 1) | (delta == 3) | (delta == 4) | (delta == 5) | (delta == 9)
                            temporary[0] = temporary[0] + T.if_then_else(positive,
                                a[source * PER_FACTOR + coordinate], -a[source * PER_FACTOR + coordinate])
                        b[destination] = temporary[0] * T.float32(1.0 / math.sqrt(12))
                    for destination in T.unroll(12):
                        a[destination * PER_FACTOR + coordinate] = b[destination]
            # Exchange all group maxima together: one barrier, not one per group.
            if WARP_GROUP > 1:
                for group in T.unroll(GROUP_LOOPS):
                    maximum[0] = 0.0
                    for p in T.unroll(P):
                        maximum[0] = T.max(maximum[0], T.abs(a[group * P + p]))
                    for stage in T.unroll(5):
                        maximum[0] = T.max(maximum[0], T.shfl_xor(maximum[0], 1 << stage))
                    if lane % 32 == 0:
                        warp_max[local_row, group * (TR // 32) + lane // 32] = maximum[0]
                T.sync_threads()
            for group in T.unroll(GROUP_LOOPS):
                maximum[0] = 0.0
                if WARP_GROUP > 1:
                    for w in T.unroll(WARP_GROUP):
                        maximum[0] = T.max(maximum[0], warp_max[local_row,
                            group * (TR // 32) + (lane // LANES) * WARP_GROUP + w])
                else:
                    for p in T.unroll(P):
                        maximum[0] = T.max(maximum[0], T.abs(a[group * P + p]))
                    for stage in T.unroll(5):
                        maximum[0] = T.max(maximum[0], T.shfl_xor(maximum[0], 1 << stage))
                temporary[0] = T.call_extern('float32', '__fmul_rn', maximum[0], T.float32(.9))
                temporary[0] = T.call_extern('float32', '__fmul_rn', temporary[0], T.float32(1.0 / 7))
                scale[0] = T.cast(T.cast(T.if_then_else(maximum[0] == 0, 1.0,
                    T.max(temporary[0], T.float32(1.1754943508222875e-38))), T.bfloat16), T.float32)
                if row < M and lane % LANES == 0:
                    Scale[row, (group * P * TR + lane) // 128] = T.cast(scale[0], T.bfloat16)
                for p in T.unroll(P):
                    temporary[0] = T.call_extern('float32', '__fdiv_rn', a[group * P + p], scale[0])
                    code[0] = T.max(-7, T.min(7, T.call_extern('int32', '__float2int_rn', temporary[0])))
                    neighbor_code[0] = T.shfl_xor(code[0], 1)
                    if row < M and lane % 2 == 0:
                        Packed[row, ((group * P + p) * TR + lane) // 2] = T.cast(
                            (code[0] & 15) | ((neighbor_code[0] & 15) << 4), T.int8)
    return tilelang.compile(kernel, pass_configs={'tl.enable_fast_math': False})


@lru_cache(None)
def best_config(M, K):
    path = Path(__file__).with_name('rht_a4_configs.json')
    if path.exists():
        data = json.loads(path.read_text())
        key = f'{32 if M <= 32 else 4096},{K}'
        if key in data:
            return Config(**data[key])
    return Config()


def rht_quant_pack(x, signs, *, out_packed, out_scale, config=None):
    if x.ndim != 2 or x.dtype != torch.bfloat16 or x.shape[1] not in (4096, 12288) or x.shape[0] < 1:
        raise ValueError('Expected BF16 X[M,K], M>0, K=4096/12288')
    m, k = x.shape
    if signs.shape != (k,) or signs.dtype != torch.int8:
        raise ValueError('Expected INT8 signs[K]')
    if out_packed.shape != (m, k // 2) or out_packed.dtype != torch.int8:
        raise ValueError('Expected INT8 output[M,K/2]')
    if out_scale.shape != (m, k // 128) or out_scale.dtype != torch.bfloat16:
        raise ValueError('Expected BF16 scales[M,K/128]')
    if any(not t.is_cuda or not t.is_contiguous() or t.device != x.device for t in [x, signs, out_packed, out_scale]):
        raise ValueError('All tensors must be contiguous on one CUDA device')
    compile_kernel(k, config or best_config(m, k))(x, signs, out_packed, out_scale)
    return out_packed, out_scale
