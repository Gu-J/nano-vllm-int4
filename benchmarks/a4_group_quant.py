"""Evaluation-only grouped A4 with fixed RHT-GPTQ weights and BF16 scales.

For grouped A, BF16-scale times INT4 is represented exactly in FP16 whenever
its exponent fits. Every generated batch checks this property. cuBLAS uses
FP32 accumulation/output; no BF16 dequantization rounding is introduced.
This is a fake-quant backend, not a native grouped INT4 inference kernel.
"""
import ctypes
import gc
import math

import torch
from torch import nn

import rht_fake_quant as old_backend
import rht_gptq as gptq
import rht_reference as ref


_GEMM = None
_H12 = None
_GRAPHS = {}


def half_gemm(x, w):
    global _GEMM
    assert x.dtype == w.dtype == torch.float16
    assert x.is_contiguous() and w.is_contiguous() and x.shape[1] == w.shape[1]
    if _GEMM is None:
        _GEMM = old_backend.CodeGemm()
    gemm = _GEMM
    m, k = x.shape
    n = w.shape[0]
    output = torch.empty((m, n), device=x.device, dtype=torch.float32)
    stream = torch.cuda.current_stream().cuda_stream
    if stream != gemm.stream:
        gemm.check(gemm.lib.cublasSetStream_v2(gemm.handle, ctypes.c_void_p(stream)))
        gemm.stream = stream
    # CUDA_R_16F=2, CUDA_R_32F=0, CUBLAS_COMPUTE_32F=68.
    gemm.check(gemm.lib.cublasGemmEx(
        gemm.handle, 1, 0, n, m, k, ctypes.byref(gemm.alpha),
        w.data_ptr(), 2, k, x.data_ptr(), 2, k, ctypes.byref(gemm.beta),
        output.data_ptr(), 0, n, 68, -1))
    return output


def rotate(x, signs):
    global _H12
    if _H12 is None:
        _H12 = ref.hadamard12(x.device)
    x = x.float() * signs
    if x.shape[-1] == 4096:
        return ref.fwht(x)
    if x.shape[-1] == 12288:
        return (torch.matmul(_H12.T, ref.fwht(x.reshape(-1, 12, 1024))) /
                math.sqrt(12)).reshape(x.shape)
    raise ValueError(x.shape)


def quantize_rotated(x, group_size, clip):
    assert group_size in (-1, 128, 64, 32) and 0 < clip <= 1
    x = x.float()
    m, k = x.shape
    size = k if group_size == -1 else group_size
    grouped = x.reshape(m, k // size, size)
    maximum = grouped.abs().amax(-1)
    scale = torch.where(maximum == 0, 1.0,
                        (maximum * clip / 7).clamp_min(torch.finfo(torch.bfloat16).tiny)).bfloat16()
    codes = (grouped / scale.float().unsqueeze(-1)).round().clamp(-7, 7)
    return codes, scale


def reconstruct(x, group_size, clip):
    codes, scale = quantize_rotated(x, group_size, clip)
    return (codes * scale.float().unsqueeze(-1)).reshape(x.shape)


def activation_eager(x, signs, group_size, clip):
    rotated = rotate(x, signs)
    codes, scale = quantize_rotated(rotated, group_size, clip)
    if group_size == -1:
        return codes.reshape(x.shape).half(), scale.reshape(-1), torch.zeros((), device=x.device, dtype=torch.int64)
    dequantized = (codes * scale.float().unsqueeze(-1)).reshape(x.shape)
    values = dequantized.half()
    failures = torch.count_nonzero((values.float() != dequantized) | ~torch.isfinite(dequantized))
    return values, scale, failures


class ActivationGraph:
    def __init__(self, x, signs, group_size, clip):
        self.x = torch.empty_like(x)
        self.signs = torch.empty_like(signs)
        self.x.copy_(x)
        self.signs.copy_(signs)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                activation_eager(self.x, self.signs, group_size, clip)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.value = activation_eager(self.x, self.signs, group_size, clip)

    def __call__(self, x, signs):
        self.x.copy_(x)
        self.signs.copy_(signs)
        self.graph.replay()
        return self.value


def activation(x, signs, group_size, clip):
    if x.shape[0] > 64:
        return activation_eager(x, signs, group_size, clip)
    key = (tuple(x.shape), x.device.index, group_size, clip)
    if key not in _GRAPHS:
        _GRAPHS[key] = ActivationGraph(x, signs, group_size, clip)
    return _GRAPHS[key](x, signs)


class SharedActivation:
    def __init__(self, signs, consumers, key, audit):
        self.signs = signs.float().contiguous()
        self.consumers, self.key, self.audit = consumers, key, audit
        self.group_size, self.clip = -1, 1.0
        self.remaining = 0
        self.last_x = self.value = None

    def __call__(self, x):
        if self.remaining == 0:
            self.last_x = x
            values, scale, failures = activation(x.reshape(-1, x.shape[-1]).contiguous(),
                                                self.signs, self.group_size, self.clip)
            if self.group_size != -1:
                self.audit.add_(failures)
            self.value = values, scale
            self.remaining = self.consumers
        elif self.last_x is not x:
            raise RuntimeError("Unexpected shared activation input")
        value = self.value
        self.remaining -= 1
        if self.remaining == 0:
            self.last_x = self.value = None
        return value


class ExperimentalLinear(nn.Module):
    def __init__(self, old, shared):
        super().__init__()
        self.register_buffer("codes", old.codes.half().contiguous())
        self.register_buffer("scale", old.scale)
        self.shared = shared
        self.in_features, self.out_features = old.in_features, old.out_features
        self.bias = None

    def forward(self, x):
        values, scale = self.shared(x)
        output = half_gemm(values, self.codes)
        if self.shared.group_size == -1:
            output.mul_(scale.float()[:, None])
        output.mul_(self.scale.float()[None, :])
        return output.bfloat16().reshape(*x.shape[:-1], self.out_features)


def install(model):
    audit = torch.zeros((), device="cuda", dtype=torch.int64)
    count = 0
    for layer_id, layer in enumerate(model.model.layers):
        for parent_name, names, group in gptq.GROUPS:
            parent = getattr(layer, parent_name)
            first = getattr(parent, names[0])
            assert isinstance(first, gptq.GPTQLinear)
            shared = SharedActivation(first.shared.signs, len(names), f"{layer_id}.{group}", audit)
            for name in names:
                setattr(parent, name, ExperimentalLinear(getattr(parent, name), shared))
                count += 1
    assert count == 252
    return audit


def configure(model, group_size, clip=1.0, per_group=None):
    _GRAPHS.clear()
    gc.collect()
    seen = set()
    for module in model.modules():
        if isinstance(module, ExperimentalLinear) and id(module.shared) not in seen:
            shared = module.shared
            assert shared.remaining == 0
            shared.group_size = group_size
            shared.clip = float(per_group[shared.key] if per_group is not None else clip)
            seen.add(id(shared))
    assert len(seen) == 144
    torch.cuda.empty_cache()


@torch.no_grad()
def self_test():
    torch.manual_seed(20261108)
    results = []
    for k in (4096, 12288):
        x = torch.randn(17, k, device="cuda", dtype=torch.bfloat16)
        x[0].zero_()
        x[1].zero_()
        x[1, 7] = 500
        signs = ref.make_signs(k, 2026, "cuda").float()
        rotated = ref.rht(x, signs)
        torch.testing.assert_close(rotate(x, signs), rotated, rtol=0, atol=0)
        weight = torch.randint(-7, 8, (64, k), device="cuda", dtype=torch.int8)
        weight_scale = (torch.rand(64, device="cuda") * .01).bfloat16()
        for g in (-1, 128, 64, 32):
            for clip in (1., .95, .9, .85):
                values, scale, failures = activation(x, signs, g, clip)
                assert failures.item() == 0
                eager = activation_eager(x, signs, g, clip)
                for a, b in zip((values, scale, failures), eager):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                output = half_gemm(values, weight.half())
                if g == -1:
                    output.mul_(scale.float()[:, None])
                    if clip == 1:
                        old_q, old_s = old_backend.activation_codes(x, signs)
                        expected = old_backend.code_gemm(old_q, weight.bfloat16()) * old_s.float()[:, None]
                        torch.testing.assert_close(output, expected, rtol=0, atol=0)
                output.mul_(weight_scale.float()[None, :])
                # Independent FP64 group dot products, including BF16 scale rounding.
                size = k if g == -1 else g
                r = rotated.double().reshape(17, k // size, size)
                maximum = r.abs().amax(-1)
                s = torch.where(maximum == 0, 1., maximum * clip / 7).bfloat16().double()
                q = (r / s[..., None]).round().clamp(-7, 7)
                wg = weight.double().reshape(64, k // size, size)
                expected = (torch.einsum("mgk,ngk->mng", q, wg) * s[:, None, :]).sum(-1)
                expected.mul_(weight_scale.double()[None, :])
                error = (output.double() - expected).square().mean().sqrt()
                relative = float(error / expected.square().mean().sqrt().clamp_min(1e-12))
                assert relative < 2e-6, (k, g, clip, relative)
                results.append({"K": k, "group_size": g, "clip": clip, "relative_rmse_vs_fp64": relative,
                                "fp16_dequantization_exact": True})
        # Updating a graph's signs must not reuse the prior rotation.
        other = ref.make_signs(k, 2027, "cuda").float()
        actual = activation(x, other, 64, .9)
        expected = activation_eager(x, other, 64, .9)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    tiny = torch.full((1, 4096), 1e-9, device="cuda")
    signs = ref.make_signs(4096, 2026, "cuda").float()
    assert activation_eager(tiny, signs, 32, 1.)[2].item() > 0, "Exponent-loss guard must reject inexact FP16"
    _GRAPHS.clear()
    return {"passed": True, "checks": results, "exponent_loss_guard": True,
            "backend": "FP16-stored exact A dequantization / integer W codes, cuBLAS FP32 accumulation, BF16 output"}
