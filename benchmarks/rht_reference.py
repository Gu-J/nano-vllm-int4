"""PyTorch reference for row-wise RHT and the existing W4A4 storage contract.

W is [N,K]. R = D H, with H normalized and orthogonal. X R and W R
preserve X W.T before quantization. For K=12*2**n, H=H12 tensor H(2**n).
This is a correctness reference, not an optimized inference implementation.
"""

import math
import torch


LAYOUT_VERSION = "row_DH_paley12_kron_sylvester_v1"


def hadamard12(device="cpu", dtype=torch.float32):
    """Paley construction over F_11. This matrix need not be symmetric."""
    residues = {i * i % 11 for i in range(1, 11)}
    h = torch.ones((12, 12), dtype=torch.int8)
    for i in range(11):
        for j in range(11):
            delta = (j - i) % 11
            h[i + 1, j + 1] = -1 if delta == 0 else (1 if delta in residues else -1)
    return h.to(device=device, dtype=dtype)


def fwht(x):
    """Normalized Sylvester transform of the last dimension, without mutation."""
    n = x.shape[-1]
    if n < 1 or n & (n - 1):
        raise ValueError("FWHT requires a power-of-two final dimension")
    shape = x.shape
    y = x
    step = 1
    while step < n:
        z = y.reshape(*shape[:-1], n // (2 * step), 2, step)
        a, b = z[..., 0, :], z[..., 1, :]
        y = torch.stack((a + b, a - b), dim=-2).reshape(shape)
        step *= 2
    return y / math.sqrt(n)


def hadamard(x, transpose=False):
    k = x.shape[-1]
    x = x if x.dtype == torch.float64 else x.float()
    if k > 0 and not (k & (k - 1)):
        return fwht(x)
    n = k // 12
    if k % 12 or n < 1 or n & (n - 1):
        raise ValueError("Supported K: 2**n or 12*2**n")
    z = fwht(x.reshape(*x.shape[:-1], 12, n))
    h = hadamard12(x.device, x.dtype)
    # Rows are flattened as [12, n]; right multiplication by H12 acts
    # on this leading factor using H12.T @ z.
    return (torch.matmul(h if transpose else h.T, z) / math.sqrt(12)).reshape(x.shape)


def make_signs(k, seed, device="cpu"):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randint(0, 2, (k,), generator=generator, dtype=torch.int8) * 2 - 1).to(device)


def rht(x, signs, inverse=False):
    if signs.shape != (x.shape[-1],):
        raise ValueError("One fixed sign is required per input channel")
    x = x if x.dtype == torch.float64 else x.float()
    signs = signs.to(device=x.device, dtype=x.dtype)
    return hadamard(x, transpose=True) * signs if inverse else hadamard(x * signs)


def quantize_rows(x):
    """Per-row absmax RTN; BF16 scale rounding precedes quantization."""
    x = x.float()
    maximum = x.abs().amax(dim=-1)
    scale = torch.where(maximum == 0, 1.0,
                        (maximum / 7).clamp_min(torch.finfo(torch.bfloat16).tiny)).bfloat16()
    q = (x / scale.float().unsqueeze(-1)).round().clamp(-7, 7).to(torch.int8)
    return q.contiguous(), scale.contiguous()


def pack_int4(q):
    if q.dtype != torch.int8 or q.shape[-1] % 2:
        raise ValueError("INT8 codes with even K are required")
    return ((q[..., ::2] & 15) | ((q[..., 1::2] & 15) << 4)).contiguous()


def unpack_int4(packed):
    p = packed.to(torch.int16) & 255
    lo, hi = p & 15, (p >> 4) & 15
    lo, hi = torch.where(lo >= 8, lo - 16, lo), torch.where(hi >= 8, hi - 16, hi)
    return torch.stack((lo, hi), dim=-1).flatten(-2).to(torch.int8)


def w4a4_reference(qx, sx, qw, sw):
    # For tested K, INT4 products/sums fit exactly in FP32, with TF32 disabled.
    integer = qx.float() @ qw.float().T
    return (integer * sx.float()[:, None] * sw.float()[None, :]).bfloat16()


def error_stats(actual, reference):
    actual, reference = actual.float(), reference.float()
    d = actual - reference
    sse = d.square().sum(dtype=torch.float64).item()
    energy = reference.square().sum(dtype=torch.float64).item()
    denom = max(energy, 1e-30)
    cosine = (actual.flatten().double() @ reference.flatten().double()).item()
    cosine /= max(math.sqrt(actual.square().sum(dtype=torch.float64).item() * energy), 1e-30)
    return {"relative_rmse": math.sqrt(sse / denom), "max_abs": d.abs().max().item(),
            "cosine": cosine, "sse": sse, "reference_energy": energy, "elements": d.numel()}


def self_test(device="cpu"):
    torch.manual_seed(2026)
    h12 = hadamard12(dtype=torch.float64)
    torch.testing.assert_close(h12 @ h12.T, torch.eye(12, dtype=torch.float64) * 12, rtol=0, atol=0)
    for k in [16, 24, 48]:
        n = k if k == 16 else k // 12
        hs = torch.ones((1, 1), dtype=torch.float64)
        while hs.shape[0] < n:
            hs = torch.cat((torch.cat((hs, hs), 1), torch.cat((hs, -hs), 1)), 0)
        h = hs if k == 16 else torch.kron(h12, hs)
        h /= math.sqrt(k)
        signs = make_signs(k, 100 + k)
        x = torch.randn(3, k, dtype=torch.float64)
        torch.testing.assert_close(rht(x, signs), (x * signs) @ h, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(rht(rht(x, signs), signs, inverse=True), x, rtol=1e-12, atol=1e-12)
    checks = []
    for k in [4096, 12288]:
        signs = make_signs(k, k + 2026, device)
        x = torch.randn(17, k, device=device)
        x[0].zero_()
        x[1].zero_()
        x[1, 7] = 1000
        w = torch.randn(32, k, device=device) / math.sqrt(k)
        xr, wr = rht(x, signs), rht(w, signs)
        inverse_error = error_stats(rht(xr, signs, inverse=True), x)
        norm_error = ((xr.norm(dim=-1) - x.norm(dim=-1)).abs() / x.norm(dim=-1).clamp_min(1)).max().item()
        linear_error = error_stats(xr @ wr.T, x @ w.T)
        assert inverse_error["relative_rmse"] < 5e-6 and norm_error < 5e-6
        assert linear_error["relative_rmse"] < 5e-6
        q, s = quantize_rows(xr)
        torch.testing.assert_close(unpack_int4(pack_int4(q)), q, rtol=0, atol=0)
        assert s.dtype == torch.bfloat16 and s[0].item() == 1 and q[0].count_nonzero().item() == 0
        checks.append({"K": k, "inverse": inverse_error, "norm_relative_max": norm_error, "linear": linear_error})
    codes = torch.arange(-8, 8, device=device, dtype=torch.int8).reshape(2, 8)
    torch.testing.assert_close(unpack_int4(pack_int4(codes)), codes, rtol=0, atol=0)
    return checks
