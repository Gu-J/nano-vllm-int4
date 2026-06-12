import torch
from torch import nn
import tilelang
import tilelang.language as T


# ==========================
# TileLang Kernel
# ==========================

@tilelang.jit
def tl_dequant_matmul_w4a16(A, B, C, scales, qzeros,
                            BLOCK_M: int, BLOCK_N: int, BLOCK_K: int,
                            SPLIT_K: int, group_size: int):
    M, N, K = T.const("M, N, K")
    A_dtype = T.bfloat16
    A: T.Tensor((M, K), A_dtype)
    B: T.Tensor((K // 8, N), T.int32)
    C: T.Tensor((M, N), T.bfloat16)
    scales: T.Tensor((K // group_size, N), T.float16)
    qzeros: T.Tensor((K // group_size, N // 8), T.int32)
    accum_dtype = T.float32
    splitK = K // SPLIT_K

    with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), SPLIT_K, threads=128) as (bx, by, bz):
        A_shared      = T.alloc_shared((BLOCK_M, BLOCK_K), A_dtype)
        B_shared      = T.alloc_shared((BLOCK_K // 8, BLOCK_N), T.int32)
        scale_shared  = T.alloc_shared((BLOCK_N,), T.bfloat16)
        zero_shared   = T.alloc_shared((BLOCK_N // 8,), T.int32)
        B_dequantized = T.alloc_fragment((BLOCK_K, BLOCK_N), A_dtype)
        C_local       = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
        C_local_bf16       = T.alloc_fragment((BLOCK_M, BLOCK_N), A_dtype)

        T.clear(C_local)

        for ko in T.Pipelined(T.ceildiv(splitK, BLOCK_K), num_stages=0):
            k_start  = bz * splitK + ko * BLOCK_K
            group_id = k_start // group_size

            T.copy(A[by * BLOCK_M, k_start], A_shared)
            T.copy(B[k_start // 8, bx * BLOCK_N], B_shared)
            T.copy(scales[group_id, bx * BLOCK_N], scale_shared)
            T.copy(qzeros[group_id, bx * BLOCK_N // 8], zero_shared)

            for i, j in T.Parallel(BLOCK_K, BLOCK_N):
                packed_w  = B_shared[i // 8, j]
                weight_i4 = (packed_w >> ((i % 8) * 4)) & 0xF

                packed_z  = zero_shared[j // 8]
                zero_i4   = (packed_z >> ((j % 8) * 4)) & 0xF

                tmp_u16  = T.cast(weight_i4+15-zero_i4, "uint16") | T.uint16(0x4300)
                tmp_fp   = T.reinterpret(tmp_u16, dtype="bfloat16") - T.bfloat16(128.0+16.0)
                
                B_dequantized[i, j] = tmp_fp * scale_shared[j]

            T.gemm(A_shared, B_dequantized, C_local)
        T.copy(C_local, C_local_bf16)
        T.atomic_add(C[by * BLOCK_M, bx * BLOCK_N], C_local_bf16)


# @tilelang.jit
# def tl_dequant_matmul_w4a16(
#     A, B, C, scales, qzeros,
#     BLOCK_M: int, BLOCK_N: int, BLOCK_K: int, SPLIT_K: int, group_size: int):
#     M, N, K = T.const("M, N, K")
#     A_dtype = T.bfloat16
#     A: T.Tensor((M, K), A_dtype)
#     B: T.Tensor((K // 8, N), T.int32)
#     C: T.Tensor((M, N), T.bfloat16)
#     scales: T.Tensor((K // group_size, N), T.float16)
#     qzeros: T.Tensor((K // group_size, N // 8), T.int32)
#     accum_dtype = T.float32

#     with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=128) as (bx, by):
#         A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), A_dtype)
#         B_shared = T.alloc_shared((BLOCK_K // 8, BLOCK_N), T.int32)
#         B_dequantized = T.alloc_shared((BLOCK_K, BLOCK_N), A_dtype)
#         C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
#         T.clear(C_local)

#         for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=3):
#             k_start = ko * BLOCK_K
#             T.copy(A[by * BLOCK_M, k_start], A_shared)
#             T.copy(B[k_start // 8, bx * BLOCK_N], B_shared)

#             for i, j in T.Parallel(BLOCK_K, BLOCK_N):
#                 packed_w = B_shared[i // 8, j]
#                 weight_i4 = (packed_w >> ((i % 8) * 4)) & 0xF
#                 B_dequantized[i, j] = T.cast(weight_i4, A_dtype)

#             T.gemm(A_shared, B_dequantized, C_local)

#         T.copy(C_local, C[by * BLOCK_M, bx * BLOCK_N])



class W4A16Linear(nn.Module):

    def __init__(self, in_features, out_features, group_size=128):
        super().__init__()

        assert group_size == 128

        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        self.qweight = nn.Parameter(
            torch.empty(in_features // 8, out_features, dtype=torch.int32),
            requires_grad=False
        )

        self.qzeros = nn.Parameter(
            torch.empty(in_features // group_size,
                        out_features // 8,
                        dtype=torch.int32),
            requires_grad=False
        )

        self.scales = nn.Parameter(
            torch.empty(in_features // group_size,
                        out_features,
                        dtype=torch.float16),
            requires_grad=False
        )

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, self.in_features)

        out = torch.zeros(
            (x.shape[0], self.out_features),
            device=x.device,
            dtype=torch.bfloat16
        )

        tl_dequant_matmul_w4a16(
            x,
            self.qweight,
            out,
            self.scales,
            self.qzeros,
            16, 64, 32, 8,
            self.group_size
        )

        return out.reshape(*shape[:-1], self.out_features)