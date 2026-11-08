"""GPTQ in the existing RHT basis, using per-channel BF16 scales and [-7,7].

The blocked second-order update follows the GPTQ algorithm (Frantar et al.,
arxiv:2210.17323). The codebook, BF16 scale rounding, RHT layout and fake GEMM
are deliberately shared with this repository's previous RTN experiment.
"""
import json
import math
from pathlib import Path

import torch
from torch import nn

import rht_reference as ref
import rht_fake_quant as backend
from validate_rht import sha256

GROUPS = [("self_attn", ["q_proj", "k_proj", "v_proj"], "qkv"),
          ("self_attn", ["o_proj"], "o"),
          ("mlp", ["gate_proj", "up_proj"], "gate_up"),
          ("mlp", ["down_proj"], "down")]
FORMAT = "rht_gptq_perchannel_bf16_scale_i4_v1"
_ROTATION_GRAPHS = {}
_H12 = {}


def rotation16_eager(x, signs, h12):
    signed = x.float() * signs
    if x.shape[-1] == 4096:
        result = ref.fwht(signed)
    elif x.shape[-1] == 12288:
        result = (torch.matmul(h12.T, ref.fwht(signed.reshape(-1, 12, 1024))) /
                  math.sqrt(12)).reshape(x.shape)
    else:
        raise ValueError(f"Unsupported RHT size: {x.shape[-1]}")
    return result.bfloat16()


class RotationGraph16:
    def __init__(self, x, signs, h12):
        self.input = torch.empty_like(x)
        self.signs = torch.empty_like(signs)
        self.input.copy_(x)
        self.signs.copy_(signs)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                rotation16_eager(self.input, self.signs, h12)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.value = rotation16_eager(self.input, self.signs, h12)

    def __call__(self, x, signs):
        self.input.copy_(x)
        self.signs.copy_(signs)
        self.graph.replay()
        return self.value


def rotation16(x, signs):
    key = x.device.index
    if key not in _H12:
        _H12[key] = ref.hadamard12(x.device)
    h12 = _H12[key]
    if x.shape[0] > 64:
        return rotation16_eager(x, signs, h12)
    key = (tuple(x.shape), x.device.index)
    if key not in _ROTATION_GRAPHS:
        _ROTATION_GRAPHS[key] = RotationGraph16(x, signs, h12)
    return _ROTATION_GRAPHS[key](x, signs)


class SharedActivation:
    def __init__(self, signs, consumers, bits=16):
        self.signs = signs.float().contiguous()
        self.consumers = consumers
        self.bits = bits
        self.remaining = 0
        self.last_x = self.value = None

    def __call__(self, x):
        if self.remaining == 0:
            flat = x.reshape(-1, x.shape[-1]).contiguous()
            self.value = (backend.activation_codes(flat, self.signs) if self.bits == 4
                          else (rotation16(flat, self.signs), None))
            self.last_x = x
            self.remaining = self.consumers
        elif self.last_x is not x:
            raise RuntimeError("Unexpected QKV/Gate-Up activation sharing")
        value = self.value
        self.remaining -= 1
        if self.remaining == 0:
            self.last_x = self.value = None
        return value


class GPTQLinear(nn.Module):
    def __init__(self, codes, scale, shared):
        super().__init__()
        assert codes.dtype == torch.int8 and scale.dtype == torch.bfloat16
        assert scale.shape == (codes.shape[0],)
        self.register_buffer("codes", codes.bfloat16().contiguous())
        self.register_buffer("scale", scale.contiguous())
        self.shared = shared
        self.in_features, self.out_features = codes.shape[1], codes.shape[0]
        self.bias = None

    def forward(self, x):
        values, activation_scale = self.shared(x)
        output = backend.code_gemm(values, self.codes)
        if activation_scale is not None:
            output.mul_(activation_scale.float()[:, None])
        output.mul_(self.scale.float()[None, :])
        return output.bfloat16().reshape(*x.shape[:-1], self.out_features)


def set_activation_bits(model, bits):
    assert bits in (4, 16)
    count = 0
    for module in model.modules():
        if isinstance(module, GPTQLinear):
            assert module.shared.remaining == 0
            module.shared.bits = bits
            count += 1
    assert count == 252
    return count


@torch.no_grad()
def factor_hessian(hessian, damping=0.01, act_order=True):
    matrix = (hessian.float() + hessian.float().T) * 0.5
    assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]
    assert torch.isfinite(matrix).all()
    dead = matrix.diagonal() <= 0
    matrix.diagonal()[dead] = 1
    order = (matrix.diagonal().argsort(descending=True) if act_order else
             torch.arange(matrix.shape[0], device=matrix.device))
    matrix = matrix.index_select(0, order).index_select(1, order)
    added = damping * matrix.diagonal().mean()
    matrix.diagonal().add_(added)
    lower = torch.linalg.cholesky(matrix)
    inverse = torch.cholesky_inverse(lower)
    upper = torch.linalg.cholesky(inverse, upper=True)
    return {"upper": upper, "order": order, "inverse_order": order.argsort(), "dead": dead,
            "damping": float(added.item()), "dead_columns": int(dead.sum().item()),
            "act_order": act_order}


@torch.no_grad()
def quantize_weight(weight, factor, scale, block_size=128):
    """Fixed per-row scales; compensate errors in still-unquantized columns."""
    assert weight.dtype == torch.float32 and scale.dtype == torch.bfloat16
    assert scale.shape == (weight.shape[0],) and torch.all(scale > 0)
    work = weight.clone()
    work[:, factor["dead"]] = 0
    work = work.index_select(1, factor["order"]).contiguous()
    upper = factor["upper"]
    codes = torch.empty_like(work, dtype=torch.int8)
    scale32 = scale.float()
    losses = torch.zeros(weight.shape[0], device=weight.device, dtype=torch.float32)
    for start in range(0, weight.shape[1], block_size):
        end = min(start + block_size, weight.shape[1])
        tile = work[:, start:end].clone()
        errors = torch.empty_like(tile)
        for column in range(end - start):
            current = tile[:, column]
            integer = (current / scale32).round().clamp(-7, 7)
            residual = (current - integer * scale32) / upper[start + column, start + column]
            codes[:, start + column] = integer.to(torch.int8)
            errors[:, column] = residual
            losses.add_(residual.square())
            tile[:, column:] -= residual[:, None] * upper[start + column, start + column:end][None, :]
        if not torch.isfinite(tile).all():
            raise FloatingPointError(f"GPTQ produced nonfinite values in block {start}")
        if end < weight.shape[1]:
            work[:, end:] -= errors @ upper[start:end, end:]
    codes = codes.index_select(1, factor["inverse_order"]).contiguous()
    return codes, {"damped_quadratic_loss": 0.5 * losses.double().sum().item(),
                   "code_min": int(codes.min().item()), "code_max": int(codes.max().item())}


class InputHessian:
    def __init__(self, k, signs):
        self.signs = signs.float().contiguous()
        self.sum = torch.zeros((k, k), device=signs.device, dtype=torch.float32)
        self.tokens = 0
        self.sample = None

    @torch.no_grad()
    def add(self, x):
        values = rotation16(x.reshape(-1, x.shape[-1]).contiguous(), self.signs).float()
        self.sum.addmm_(values.T, values)
        self.tokens += values.shape[0]
        if self.sample is None:
            indices = torch.linspace(0, values.shape[0] - 1, min(128, values.shape[0]),
                                     device=values.device).long()
            self.sample = values.index_select(0, indices).clone()

    def finish(self):
        assert self.tokens > 0
        self.sum.mul_(2 / self.tokens)
        return self.sum


def load_layer(layer, layer_index, directory, manifest, bits=16):
    from safetensors.torch import load_file
    entry = manifest["layers"][str(layer_index)]
    path = directory / entry["file"]
    assert sha256(path) == entry["sha256"]
    state = load_file(str(path), device="cpu")
    for group, (parent_name, names, group_name) in enumerate(GROUPS):
        parent = getattr(layer, parent_name)
        k = getattr(parent, names[0]).in_features
        signs = state[f"signs.{group_name}"]
        expected = ref.make_signs(k, manifest["config"]["seed"] + layer_index * 17 + group)
        assert torch.equal(signs, expected)
        shared = SharedActivation(signs.cuda(), len(names), bits)
        for name in names:
            old = getattr(parent, name)
            key = f"{parent_name}.{name}"
            packed, scale = state[key + ".packed"], state[key + ".scale"]
            assert packed.dtype == torch.int8 and scale.dtype == torch.bfloat16
            assert packed.shape == (old.out_features, old.in_features // 2)
            assert scale.shape == (old.out_features,) and torch.isfinite(scale).all() and torch.all(scale > 0)
            codes = ref.unpack_int4(packed.cuda())
            assert codes.min() >= -7 and codes.max() <= 7
            setattr(parent, name, GPTQLinear(codes, scale.cuda(), shared))
    return 7


@torch.no_grad()
def load_checkpoint(model, directory, bits=16):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["format"] == FORMAT and manifest["complete"]
    assert set(manifest["layers"]) == {str(i) for i in range(36)}
    for name, digest in manifest["config"]["source_sha256"].items():
        assert sha256(Path(__file__).with_name(name)) == digest, name
    count = sum(load_layer(layer, i, directory, manifest, bits)
                for i, layer in enumerate(model.model.layers))
    assert count == 252
    torch.cuda.empty_cache()
    return {"projections": count, "layers": 36, "activation_bits": bits, "format": FORMAT,
            "checkpoint_manifest_sha256": sha256(directory / "manifest.json"),
            "calibration": manifest["config"], "weights_shared_between_a16_and_a4": True,
            "scales": "BF16 per-output-channel W; BF16 per-token A when A4",
            "other_components": "BF16 embedding, LM head, norms, attention and KV cache"}


@torch.no_grad()
def check_reload(model, directory):
    from safetensors.torch import load_file
    state = load_file(str(Path(directory) / "reload_smoke.safetensors"))
    actual = model(state["input_ids"].cuda(), use_cache=False, logits_to_keep=1).logits.cpu()
    torch.testing.assert_close(actual, state["logits"], rtol=0, atol=0)
    return {"reload_logits_exact": True, "input_tokens": state["input_ids"].numel()}


def self_test():
    """Independent shrinking-inverse reference plus storage/runtime checks."""
    torch.manual_seed(2026)
    checks = []
    for act_order in (False, True):
        for block in (1, 5, 128):
            x = torch.randn(70, 13, dtype=torch.float64)
            x[:, 2] += 2 * x[:, 0]
            x[:, 9] = 0
            h = (2 * x.T @ x / x.shape[0]).float()
            w = torch.randn(7, 13).float()
            w[0] = 0
            _, scales = ref.quantize_rows(w)
            factor = factor_hessian(h, act_order=act_order)
            actual, _ = quantize_weight(w, factor, scales, block_size=block)
            # Recompute the inverse of the remaining Hessian at each column.
            matrix = h.double().clone()
            matrix.diagonal()[factor["dead"]] = 1
            order = factor["order"]
            matrix = matrix[order][:, order]
            matrix.diagonal().add_(factor["damping"])
            remaining = w.double().clone()
            remaining[:, factor["dead"]] = 0
            remaining = remaining[:, order]
            expected = torch.empty_like(actual)
            for c in range(w.shape[1]):
                integer = (remaining[:, c] / scales.double()).round().clamp(-7, 7)
                expected[:, c] = integer.to(torch.int8)
                inverse = torch.linalg.inv(matrix[c:, c:])
                error = remaining[:, c] - integer * scales.double()
                remaining[:, c:] -= error[:, None] * (inverse[0] / inverse[0, 0])[None, :]
            expected = expected[:, factor["inverse_order"]]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(ref.unpack_int4(ref.pack_int4(actual)), actual, rtol=0, atol=0) if actual.shape[1] % 2 == 0 else None
            checks.append({"act_order": act_order, "block_size": block, "independent_reference_exact": True})
    w = torch.randn(7, 16)
    q, s = ref.quantize_rows(w)
    got, _ = quantize_weight(w, factor_hessian(torch.eye(16)), s)
    torch.testing.assert_close(q, got, rtol=0, atol=0)
    torch.testing.assert_close(ref.unpack_int4(ref.pack_int4(got)), got, rtol=0, atol=0)
    backend_checks = backend.backend_self_test(32)
    rotation_checks = []
    for k in (4096, 12288):
        for m in (17, 32):
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            x[0] = 0
            for seed in (2026, 2027):
                signs = ref.make_signs(k, seed, "cuda").float()
                torch.testing.assert_close(rotation16(x, signs), ref.rht(x, signs).bfloat16(), rtol=0, atol=0)
            a = torch.randint(-128, 128, (m, k), device="cuda").bfloat16() / 32
            b = torch.randint(-7, 8, (256, k), device="cuda").bfloat16()
            torch.testing.assert_close(backend.code_gemm(a, b), a.float() @ b.float().T, rtol=0, atol=0)
            rotation_checks.append({"M": m, "K": k, "rotation16_exact": True, "a16_gemm_exact_dyadic_inputs": True})
    return {"gptq": checks, "diagonal_hessian_equals_rtn": True,
            "a4_backend": backend_checks, "a16_backend": rotation_checks}
