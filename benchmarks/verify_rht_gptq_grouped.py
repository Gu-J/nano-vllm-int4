"""Independent static-group GPTQ reference and lossless fake-quant backend checks."""
import argparse
import json
import os
from pathlib import Path

from validate_rht import select_gpu, exclusive, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--output-dir", type=Path, default=Path("benchmarks/results/rht_gptq_grouped/check"))
    args = p.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["OMP_NUM_THREADS"] = "4"
    gpu = select_gpu(args.gpu)
    import torch
    import rht_gptq_grouped as q
    import rht_reference as ref
    import a4_group_quant as ab
    import rht_fake_quant as old
    torch.set_num_threads(4); torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.manual_seed(20261109)
    checks = []
    for act_order in [False, True]:
        for group_size in [4, 7, 128]:
            for block_size in [1, 5, 128]:
                x = torch.randn(70, 13, dtype=torch.float64); x[:, 2] += 2*x[:, 0]; x[:, 9] = 0
                h = (2*x.T@x/x.shape[0]).float()
                w = torch.randn(7, 13); w[0] = 0
                _, scales = q.quantize_groups(w, group_size)
                factor = q.factor_hessian(h, act_order=act_order)
                actual, _ = q.quantize_weight(w, factor, scales, block_size, group_size)
                order = factor["order"]
                matrix = h.double().clone(); matrix.diagonal()[factor["dead"]] = 1
                matrix = matrix[order][:, order]; matrix.diagonal().add_(factor["damping"])
                remaining = w.double().clone(); remaining[:, factor["dead"]] = 0; remaining = remaining[:, order]
                expected = torch.empty_like(actual)
                for c in range(w.shape[1]):
                    s = scales[:, int(order[c]) // group_size].double()
                    integer = (remaining[:, c]/s).round().clamp(-7, 7)
                    expected[:, c] = integer.to(torch.int8)
                    inverse = torch.linalg.inv(matrix[c:, c:])
                    error = remaining[:, c] - integer*s
                    remaining[:, c:] -= error[:, None]*(inverse[0]/inverse[0, 0])[None, :]
                expected = expected[:, factor["inverse_order"]]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                checks.append({"act_order": act_order, "group_size": group_size, "block_size": block_size,
                               "independent_fp64_codes_exact": True})
    w = torch.randn(7, 256)
    rtn, scales = q.quantize_groups(w)
    actual, _ = q.quantize_weight(w, q.factor_hessian(torch.eye(256)), scales)
    torch.testing.assert_close(actual, rtn, rtol=0, atol=0)
    torch.testing.assert_close(ref.unpack_int4(ref.pack_int4(actual)), actual, rtol=0, atol=0)
    backend_checks = []
    for k in [4096, 12288]:
        exclusive(gpu)
        m, n = 17, 64
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16); x[0] = 0
        signs = ref.make_signs(k, 2026, "cuda").float()
        qw = torch.randint(-7, 8, (n, k), device="cuda", dtype=torch.int8)
        sw = (torch.rand(n, k//128, device="cuda")*.01+.001).bfloat16()
        weight = q.dequantize(qw, sw)
        shared = q.SharedActivation(signs, 1, bits=16)
        linear = q.GPTQLinear(qw, sw, shared)
        rotated16 = q.original.rotation16(x, signs)
        calibrated = old.code_gemm(rotated16, linear.high) + old.code_gemm(rotated16, linear.low)
        gold16 = rotated16.double() @ weight.double().T
        err16 = float((calibrated.double()-gold16).square().mean().sqrt()/gold16.square().mean().sqrt())
        assert err16 < 2e-6
        torch.testing.assert_close(linear(x), calibrated.bfloat16(), rtol=0, atol=0)
        audit = torch.zeros((), device="cuda", dtype=torch.int64)
        linear.enable_a4(); shared.bits = 4; shared.audit = audit
        value, factor, failures = q.activation_eager(x, signs)
        assert failures.item() == 0
        rotated = ref.rht(x, signs)
        aq, sa = ab.quantize_rotated(rotated, 128, .9)
        dequant_a = (aq*sa.float()[...,None]).reshape(x.shape)
        torch.testing.assert_close(value.float()*factor[:,None], dequant_a, rtol=0, atol=0)
        output = ab.half_gemm(value, linear.weight_half)*linear.weight_factor[None,:]*factor[:,None]
        gold = dequant_a.double() @ weight.double().T
        relative = float((output.double()-gold).square().mean().sqrt()/gold.square().mean().sqrt())
        assert relative < 2e-6
        # Power-of-two normalization adds no rounding for ordinary in-range data.
        unscaled = ab.half_gemm(dequant_a.half(), weight.half())
        torch.testing.assert_close(unscaled, output, rtol=0, atol=0)
        for _ in range(2):
            torch.testing.assert_close(linear(x), output.bfloat16(), rtol=0, atol=0)
        assert audit.item() == 0
        graph = q.ActivationGraph(x, signs)
        for seed in [2026, 2027]:
            signs2 = ref.make_signs(k, seed, "cuda").float()
            a, b = graph(x, signs2), q.activation_eager(x, signs2)
            for v1,v2 in zip(a,b):torch.testing.assert_close(v1,v2,rtol=0,atol=0)
        for multiplier in [1e-20, 1e20]:
            tiny_scale = (sw*multiplier).bfloat16()
            original = q.dequantize(qw, tiny_scale)
            v, f, failures = q.normalize_half(original)
            assert failures.item() == 0
            torch.testing.assert_close(v.float()*f[:,None], original, rtol=0, atol=0)
        backend_checks.append({"K": k, "a16_relative_rmse_fp64": err16, "a4_relative_rmse_fp64": relative,
                               "normalization_exact": True, "unscaled_backend_bitexact": True, "graph_sign_update_exact": True})
    report = {"complete": True, "gpu": gpu, "gptq_checks": checks, "diagonal_hessian_equals_rtn": True,
              "packing_exact": True, "backend_checks": backend_checks,
              "source_sha256": {p:sha256(Path(__file__).with_name(p)) for p in ['rht_gptq_grouped.py','verify_rht_gptq_grouped.py']}}
    (args.output_dir/'self_test.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    main()
