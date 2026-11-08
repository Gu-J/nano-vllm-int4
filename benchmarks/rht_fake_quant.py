"""Evaluation-only W4A4 fake quant, with the existing BF16 scale/output contract.

INT4 codes are stored losslessly in BF16, cuBLAS multiplies them into FP32,
then scales are applied before the sole BF16 output rounding. No INT4 inference
kernel is used here. All 252 decoder projections are quantized uniformly.
"""
import ctypes
import math
from pathlib import Path
import torch
from torch import nn
import rht_reference as ref


class CodeGemm:
    """cuBLAS BF16 inputs / FP32 accumulation and output, without code rounding."""
    def __init__(self):
        path = Path(torch.__file__).parent.parent / "nvidia/cublas/lib/libcublas.so.12"
        self.lib = ctypes.CDLL(str(path))
        ptr, integer = ctypes.c_void_p, ctypes.c_int
        self.lib.cublasCreate_v2.argtypes = [ctypes.POINTER(ptr)]
        self.lib.cublasCreate_v2.restype = integer
        self.lib.cublasSetStream_v2.argtypes = [ptr, ptr]
        self.lib.cublasSetMathMode.argtypes = [ptr, integer]
        self.lib.cublasSetPointerMode_v2.argtypes = [ptr, integer]
        self.lib.cublasGemmEx.argtypes = [ptr, integer, integer, integer, integer, integer,
                                        ptr, ptr, integer, integer, ptr, integer, integer,
                                        ptr, ptr, integer, integer, integer, integer]
        self.lib.cublasGemmEx.restype = integer
        self.handle = ptr()
        self.check(self.lib.cublasCreate_v2(ctypes.byref(self.handle)))
        self.check(self.lib.cublasSetMathMode(self.handle, 16))
        self.check(self.lib.cublasSetPointerMode_v2(self.handle, 0))
        self.alpha, self.beta = ctypes.c_float(1.0), ctypes.c_float(0.0)
        self.stream = None

    @staticmethod
    def check(status):
        if status != 0:
            raise RuntimeError(f"cuBLAS error {status}")

    def __call__(self, x, w):
        assert x.dtype == w.dtype == torch.bfloat16 and x.is_contiguous() and w.is_contiguous()
        assert x.device == w.device and x.shape[1] == w.shape[1]
        m, k = x.shape
        n = w.shape[0]
        output = torch.empty((m, n), device=x.device, dtype=torch.float32)
        stream = torch.cuda.current_stream().cuda_stream
        if stream != self.stream:
            self.check(self.lib.cublasSetStream_v2(self.handle, ctypes.c_void_p(stream)))
            self.stream = stream
        # Column-major C^T = W @ X^T, interpreting row-major storage as transposes.
        # CUDA_R_16BF=14, CUDA_R_32F=0, CUBLAS_COMPUTE_32F=68, DEFAULT algo=-1.
        self.check(self.lib.cublasGemmEx(self.handle, 1, 0, n, m, k,
                                        ctypes.byref(self.alpha), w.data_ptr(), 14, k,
                                        x.data_ptr(), 14, k, ctypes.byref(self.beta),
                                        output.data_ptr(), 0, n, 68, -1))
        return output


_GEMM = None
_H12 = None
_GRAPHS = {}


def code_gemm(x, w):
    global _GEMM
    if _GEMM is None:
        _GEMM = CodeGemm()
    return _GEMM(x, w)


def activation_codes_eager(x, signs, h12):
    x = x.float() * signs
    k = x.shape[-1]
    if k == 4096:
        rotated = ref.fwht(x)
    elif k == 12288:
        rotated = (torch.matmul(h12.T, ref.fwht(x.reshape(-1, 12, 1024))) / math.sqrt(12)).reshape(x.shape)
    else:
        raise ValueError(f"Unsupported input size {k}")
    q, scale = ref.quantize_rows(rotated)
    return q.bfloat16(), scale


class ActivationGraph:
    """Replay the PyTorch reference for small decode batches; no new RHT kernel."""
    def __init__(self, x, signs, h12):
        self.x = torch.empty_like(x)
        self.signs = torch.empty_like(signs)
        self.x.copy_(x)
        self.signs.copy_(signs)
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                activation_codes_eager(self.x, self.signs, h12)
        torch.cuda.current_stream().wait_stream(warmup_stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.q, self.scale = activation_codes_eager(self.x, self.signs, h12)

    def __call__(self, x, signs):
        self.x.copy_(x)
        self.signs.copy_(signs)
        self.graph.replay()
        return self.q, self.scale


def activation_codes(x, signs):
    global _H12
    if _H12 is None:
        _H12 = ref.hadamard12(x.device)
    if x.shape[0] > 64:
        return activation_codes_eager(x, signs, _H12)
    key = (tuple(x.shape), x.device.index)
    if key not in _GRAPHS:
        _GRAPHS[key] = ActivationGraph(x, signs, _H12)
    return _GRAPHS[key](x, signs)


class SharedActivation:
    def __init__(self, signs, consumers):
        self.signs = signs.float().contiguous()
        self.consumers = consumers
        self.last_x, self.value, self.remaining = None, None, 0

    def __call__(self, x):
        if self.remaining == 0:
            self.last_x = x
            self.value = activation_codes(x.reshape(-1, x.shape[-1]).contiguous(), self.signs)
            self.remaining = self.consumers
        elif self.last_x is not x:
            raise RuntimeError("Shared projection activation reuse changed unexpectedly")
        result = self.value
        self.remaining -= 1
        if self.remaining == 0:
            self.last_x, self.value = None, None
        return result


class FakeQuantLinear(nn.Module):
    def __init__(self, codes, scale, shared):
        super().__init__()
        self.register_buffer("codes", codes.bfloat16().contiguous())
        self.register_buffer("scale", scale.contiguous())
        self.shared = shared
        self.in_features, self.out_features = codes.shape[1], codes.shape[0]

    def forward(self, x):
        q, scale = self.shared(x)
        acc = code_gemm(q, self.codes)
        acc.mul_(scale.float()[:, None]).mul_(self.scale.float()[None, :])
        return acc.bfloat16().reshape(*x.shape[:-1], self.out_features)


@torch.no_grad()
def convert_model(model, seed=2026):
    count = 0
    shapes = {}
    for i, layer in enumerate(model.model.layers):
        groups = [(layer.self_attn, ["q_proj", "k_proj", "v_proj"], 0),
                  (layer.self_attn, ["o_proj"], 1),
                  (layer.mlp, ["gate_proj", "up_proj"], 2), (layer.mlp, ["down_proj"], 3)]
        for parent, names, group in groups:
            k = getattr(parent, names[0]).weight.shape[1]
            signs = ref.make_signs(k, seed + i * 17 + group, "cuda")
            shared = SharedActivation(signs, len(names))
            for name in names:
                old = getattr(parent, name)
                assert old.bias is None
                rotated = ref.rht(old.weight, signs)
                codes, scale = ref.quantize_rows(rotated)
                replacement = FakeQuantLinear(codes, scale, shared)
                setattr(parent, name, replacement)
                count += 1
                shapes[f"{replacement.out_features},{replacement.in_features}"] = shapes.get(f"{replacement.out_features},{replacement.in_features}", 0) + 1
                del old, rotated, codes, scale
    assert count == 252
    torch.cuda.empty_cache()
    return {"projections": count, "layers": len(model.model.layers), "groups_per_layer": 4,
            "seed": seed, "layout_version": ref.LAYOUT_VERSION, "shapes": shapes,
            "weight_and_activation_codes": "symmetric absmax RTN [-7,7]",
            "scales": "BF16 per-output-channel W / per-token A; rounded before codes",
            "accumulation": "BF16-stored exact integer codes, cuBLAS FP32 output, scales then BF16 cast",
            "other_components": "BF16 embedding, LM head, norms, attention and KV cache",
            "model_checkpoint_written": False}


@torch.no_grad()
def backend_self_test(batch_size):
    torch.manual_seed(2026)
    checks = []
    for k in [4096, 12288]:
        for m in [17, batch_size]:
            qx = torch.randint(-8, 8, (m, k), device="cuda", dtype=torch.int8)
            qw = torch.randint(-8, 8, (256, k), device="cuda", dtype=torch.int8)
            integer = code_gemm(qx.bfloat16(), qw.bfloat16())
            torch.testing.assert_close(integer, qx.float() @ qw.float().T, rtol=0, atol=0)
            sx = torch.rand(m, device="cuda").bfloat16()
            sw = torch.rand(256, device="cuda").bfloat16()
            scaled = (integer * sx.float()[:, None] * sw.float()[None, :]).bfloat16()
            torch.testing.assert_close(scaled, ref.w4a4_reference(qx, sx, qw, sw), rtol=0, atol=0)
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            x[0].zero_()
            for seed in [2026, 2027]:
                signs = ref.make_signs(k, seed, "cuda")
                q, scale = activation_codes(x, signs.float())
                expected_q, expected_scale = ref.quantize_rows(ref.rht(x, signs))
                torch.testing.assert_close(q, expected_q.bfloat16(), rtol=0, atol=0)
                torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)
            checks.append({"M": m, "K": k, "code_dot_exact": True, "scaled_output_exact": True,
                           "activation_graph_exact": True, "changed_signs_checked": True})
    return checks
