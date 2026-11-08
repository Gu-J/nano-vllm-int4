"""RHT-aware GPTQ with static original-K groups, BF16 scales, and [-7,7].

Static groups remain contiguous after undoing act-order. Evaluation uses exact
FP16 storage with power-of-two row normalization, FP32 GEMM and BF16 outputs.
The normalization is a fake-quant storage device, not an additional quantizer.
"""
import json
from pathlib import Path

import torch
from torch import nn

import rht_reference as ref
import rht_fake_quant as backend
import rht_gptq as original
import a4_group_quant as activation_backend
from validate_rht import sha256

GROUPS = original.GROUPS
GROUP_SIZE = 128
FORMAT = "rht_gptq_static_group128_bf16_scale_i4_v1"
factor_hessian = original.factor_hessian
InputHessian = original.InputHessian
_GRAPHS = {}


def quantize_groups(weight, group_size=GROUP_SIZE):
    n, k = weight.shape
    assert group_size > 0
    scales, codes = [], []
    for start in range(0, k, group_size):
        q, s = ref.quantize_rows(weight[:, start:start + group_size])
        scales.append(s); codes.append(q)
    return torch.cat(codes, dim=1).contiguous(), torch.stack(scales, dim=1).contiguous()


def dequantize(codes, scale, group_size=GROUP_SIZE):
    assert scale.shape == (codes.shape[0], (codes.shape[1] + group_size - 1) // group_size)
    return codes.float() * scale.float().repeat_interleave(group_size, dim=1)[:, :codes.shape[1]]


@torch.no_grad()
def quantize_weight(weight, factor, scale, block_size=128, group_size=GROUP_SIZE):
    assert weight.dtype == torch.float32 and scale.dtype == torch.bfloat16
    assert scale.shape == (weight.shape[0], (weight.shape[1] + group_size - 1) // group_size)
    assert torch.isfinite(scale).all() and torch.all(scale > 0)
    work = weight.clone()
    work[:, factor["dead"]] = 0
    work = work.index_select(1, factor["order"]).contiguous()
    upper = factor["upper"]
    codes = torch.empty_like(work, dtype=torch.int8)
    # Each ordered column selects its ORIGINAL-K group. Scale vectors are
    # contiguous across output channels, and the order is copied only once.
    scales32 = scale.float().T.contiguous()
    group_ids = (factor["order"] // group_size).cpu().tolist()
    losses = torch.zeros(weight.shape[0], device=weight.device)
    for start in range(0, weight.shape[1], block_size):
        end = min(start + block_size, weight.shape[1])
        tile = work[:, start:end].clone()
        errors = torch.empty_like(tile)
        for column in range(end - start):
            s = scales32[group_ids[start + column]]
            current = tile[:, column]
            integer = (current / s).round().clamp(-7, 7)
            residual = (current - integer * s) / upper[start + column, start + column]
            codes[:, start + column] = integer.to(torch.int8)
            errors[:, column] = residual
            losses.add_(residual.square())
            tile[:, column:] -= residual[:, None] * upper[start + column, start + column:end][None, :]
        if not torch.isfinite(tile).all():
            raise FloatingPointError(f"Nonfinite GPTQ block {start}")
        if end < weight.shape[1]:
            work[:, end:] -= errors @ upper[start:end, end:]
    codes = codes.index_select(1, factor["inverse_order"]).contiguous()
    return codes, {"damped_quadratic_loss": .5 * losses.double().sum().item(),
                   "code_min": int(codes.min()), "code_max": int(codes.max()),
                   "static_original_k_groups": True, "group_size": group_size}


def normalize_half(values):
    maximum = values.abs().amax(-1)
    _, exponents = torch.frexp(maximum)
    factor = torch.where(maximum == 0, torch.ones_like(maximum),
                         torch.ldexp(torch.ones_like(maximum), exponents - 1))
    stored = (values / factor[:, None]).half()
    failures = torch.count_nonzero((stored.float() * factor[:, None] != values) | ~torch.isfinite(values))
    return stored.contiguous(), factor, failures


def activation_eager(x, signs):
    rotated = activation_backend.rotate(x, signs)
    codes, scale = activation_backend.quantize_rotated(rotated, GROUP_SIZE, .9)
    values = (codes * scale.float().unsqueeze(-1)).reshape(x.shape)
    return normalize_half(values)


class ActivationGraph:
    def __init__(self, x, signs):
        self.x, self.signs = torch.empty_like(x), torch.empty_like(signs)
        self.x.copy_(x); self.signs.copy_(signs)
        stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                activation_eager(self.x, self.signs)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.value = activation_eager(self.x, self.signs)

    def __call__(self, x, signs):
        self.x.copy_(x); self.signs.copy_(signs); self.graph.replay()
        return self.value


class SharedActivation:
    def __init__(self, signs, consumers, bits=16, audit=None):
        self.signs = signs.float().contiguous()
        self.consumers, self.bits, self.audit = consumers, bits, audit
        self.remaining = 0
        self.last_x = self.value = None

    def __call__(self, x):
        if self.remaining == 0:
            flat = x.reshape(-1, x.shape[-1]).contiguous()
            if self.bits == 16:
                self.value = original.rotation16(flat, self.signs), None
            else:
                key = (tuple(flat.shape), flat.device.index)
                if flat.shape[0] > 64:
                    values, factor, failures = activation_eager(flat, self.signs)
                else:
                    if key not in _GRAPHS:
                        _GRAPHS[key] = ActivationGraph(flat, self.signs)
                    values, factor, failures = _GRAPHS[key](flat, self.signs)
                self.audit.add_(failures)
                self.value = values, factor
            self.last_x = x; self.remaining = self.consumers
        elif self.last_x is not x:
            raise RuntimeError("Unexpected QKV/Gate-Up shared input")
        value = self.value; self.remaining -= 1
        if self.remaining == 0:
            self.last_x = self.value = None
        return value


class GPTQLinear(nn.Module):
    def __init__(self, codes, scale, shared):
        super().__init__()
        assert codes.dtype == torch.int8 and scale.dtype == torch.bfloat16
        self.in_features, self.out_features = codes.shape[1], codes.shape[0]
        self.shared, self.bias = shared, None
        self.register_buffer("scale", scale.contiguous())
        weight = dequantize(codes, scale)
        # Exact two-BF16 representation for calibration with BF16 activations.
        high = weight.bfloat16()
        low = (weight - high.float()).bfloat16()
        torch.testing.assert_close(high.float() + low.float(), weight, rtol=0, atol=0)
        self.register_buffer("high", high.contiguous())
        self.register_buffer("low", low.contiguous())
        self.register_buffer("weight_half", None)
        self.register_buffer("weight_factor", None)

    def enable_a4(self):
        if self.weight_half is None:
            weight = self.high.float() + self.low.float()
            values, factor, failures = normalize_half(weight)
            assert failures.item() == 0, "Weight FP16 representation would be lossy"
            self.weight_half, self.weight_factor = values, factor
            self.high = self.low = None

    def forward(self, x):
        values, factor = self.shared(x)
        if self.shared.bits == 16:
            output = backend.code_gemm(values, self.high)
            output.add_(backend.code_gemm(values, self.low))
        else:
            output = activation_backend.half_gemm(values, self.weight_half)
            output.mul_(self.weight_factor[None, :])
            output.mul_(factor[:, None])
        return output.bfloat16().reshape(*x.shape[:-1], self.out_features)


def set_activation_bits(model, bits):
    assert bits == 4
    audit = torch.zeros((), device="cuda", dtype=torch.int64)
    count = 0
    for module in model.modules():
        if isinstance(module, GPTQLinear):
            assert module.shared.remaining == 0
            module.enable_a4()
            module.shared.bits, module.shared.audit = bits, audit
            count += 1
    assert count == 252
    _GRAPHS.clear(); original._ROTATION_GRAPHS.clear()
    torch.cuda.empty_cache()
    return audit


def load_layer(layer, layer_index, directory, manifest, bits=16):
    from safetensors.torch import load_file
    assert bits == 16
    entry = manifest["layers"][str(layer_index)]; path = directory / entry["file"]
    assert sha256(path) == entry["sha256"]
    state = load_file(str(path), device="cpu")
    for group, (parent_name, names, group_name) in enumerate(GROUPS):
        parent = getattr(layer, parent_name)
        k = getattr(parent, names[0]).in_features
        signs = state[f"signs.{group_name}"]
        assert torch.equal(signs, ref.make_signs(k, manifest["config"]["seed"] + layer_index * 17 + group))
        shared = SharedActivation(signs.cuda(), len(names), bits)
        for name in names:
            old = getattr(parent, name); key = f"{parent_name}.{name}"
            packed, scale = state[key + ".packed"], state[key + ".scale"]
            assert packed.dtype == torch.int8 and packed.shape == (old.out_features, old.in_features // 2)
            assert scale.dtype == torch.bfloat16 and scale.shape == (old.out_features, old.in_features // GROUP_SIZE)
            assert torch.isfinite(scale).all() and torch.all(scale > 0)
            codes = ref.unpack_int4(packed.cuda())
            assert codes.min() >= -7 and codes.max() <= 7
            setattr(parent, name, GPTQLinear(codes, scale.cuda(), shared))
    return 7


@torch.no_grad()
def load_checkpoint(model, directory, bits=16):
    directory = Path(directory); manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["format"] == FORMAT and manifest["complete"]
    assert manifest["config"]["group_size"] == GROUP_SIZE and manifest["config"]["static_groups"]
    assert set(manifest["layers"]) == {str(i) for i in range(36)}
    for name, digest in manifest["config"]["source_sha256"].items():
        assert sha256(Path(__file__).with_name(name)) == digest, name
    count = sum(load_layer(layer, i, directory, manifest, bits) for i, layer in enumerate(model.model.layers))
    assert count == 252
    torch.cuda.empty_cache()
    return {"projections": count, "layers": 36, "activation_bits": bits, "weight_group_size": GROUP_SIZE,
            "activation_group_size": GROUP_SIZE, "activation_clipping": .9, "format": FORMAT,
            "checkpoint_manifest_sha256": sha256(directory / "manifest.json"), "calibration": manifest["config"],
            "scales": "BF16 per-output-channel/per-K128 W, per-token/per-K128 A",
            "other_components": "BF16 embedding, LM head, norms, attention and KV cache",
            "backend": "exact normalized FP16 storage; cuBLAS FP32 accumulation/output; final BF16 rounding"}


check_reload = original.check_reload
