"""Independent references for activation AND weight G=128 native INT4 GEMM."""
from benchmark_utils import source_path
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

from validate_rht import select_gpu, exclusive, sha256
from verify_w4a4_grouped import pack, reference as a_only_reference


def reference(qa, qw, sa, sw, precise=False):
    """FP32 integer dots plus FP64 arithmetic emulate each FP32 FMA rounding.

    INT4 group dot products and the product of two BF16 scales are exact in
    FP32. For these bounded test inputs, FP64 represents the dot/scale product
    and addition accurately enough to provide an independent fused reference.
    precise=True instead retains FP64 across all groups as an error oracle.
    """
    import torch
    m, k = qa.shape
    total = torch.zeros((m, qw.shape[0]), device=qa.device, dtype=torch.float64 if precise else torch.float32)
    for g in range(k // 128):
        dot = qa[:, g * 128:(g + 1) * 128].float() @ qw[:, g * 128:(g + 1) * 128].float().T
        scale = sa[:, g].float()[:, None] * sw[:, g].float()[None, :]
        value = dot.double() * scale.double() + total.double()
        total = value if precise else value.float()
    return total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--output-dir", type=Path, default=Path("benchmarks/results/w4a4_aw_grouped/check"))
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(f"benchmarks/results/w4a4_aw_grouped/cache/gpu{args.gpu}").resolve()
    os.environ["TILELANG_CACHE_DIR"] = str(cache)
    os.environ["TILELANG_TMP_DIR"] = str(cache / "tmp")
    gpu = select_gpu(args.gpu)
    import torch
    from nanovllm.kernels import w4a4_grouped_kernel as wk
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261110)
    checks = []
    configs = [wk.GroupedConfig(32, 64, 2, 128, ac, wc) for ac in [False, True] for wc in [False, True]]
    for m, n, k in [(17, 96, 512), (33, 136, 4096), (32, 128, 12288)]:
        exclusive(gpu)
        qa = torch.randint(-8, 8, (m, k), device="cuda", dtype=torch.int8)
        qw = torch.randint(-8, 8, (n, k), device="cuda", dtype=torch.int8)
        qa[0].zero_(); qw[0].zero_()
        sa = (torch.rand(m, k // 128, device="cuda") * 2).bfloat16()
        sw = (torch.rand(n, k // 128, device="cuda") * .03).bfloat16()
        sa[1, ::2] = 0; sw[1, 1::2] = 0
        ap, wp = pack(qa), pack(qw)
        expected = reference(qa, qw, sa, sw)
        precise = reference(qa, qw, sa, sw, precise=True)
        for cfg in configs:
            storage = torch.full((m * n + 128,), -1234, device="cuda", dtype=torch.bfloat16)
            sentinel = storage[:64].clone(); out = storage[64:-64].view(m, n)
            for _ in range(3):
                out.fill_(float("nan"))
                actual = wk.w4a4_grouped_gemm(ap, wp, sa, sw, out=out, config=cfg)
                assert actual.data_ptr() == out.data_ptr()
                torch.testing.assert_close(actual, expected.bfloat16(), rtol=0, atol=0)
                torch.testing.assert_close(storage[:64], sentinel, rtol=0, atol=0)
                torch.testing.assert_close(storage[-64:], sentinel, rtol=0, atol=0)
            out32 = torch.empty((m, n), device="cuda", dtype=torch.float32)
            wk.compile_grouped_gemm(m, n, k, cfg, output_fp32=True, weight_grouped=True)(ap, wp, sa, sw, out32)
            torch.testing.assert_close(out32, expected, rtol=0, atol=0)
            rrmse = float((out32.double() - precise).square().mean().sqrt() / precise.square().mean().sqrt().clamp_min(1e-20))
            assert rrmse < 2e-6
            source = wk.compile_grouped_gemm(m, n, k, cfg, weight_grouped=True).get_kernel_source()
            assert "mma_sync<tl::DataType::kInt4, tl::DataType::kInt4, tl::DataType::kInt32" in source
            assert "atomicAdd" not in source and "atomic_add" not in source
            (args.output_dir / f"m{m}_n{n}_k{k}_a{int(cfg.cache_scales)}_w{int(cfg.cache_weight_scales)}.cu").write_text(source)
            row = {"M": m, "N": n, "K": k, "config": asdict(cfg), "bf16_exact": True,
                   "fp32_exact_fused_reference": True, "fp64_relative_rmse": rrmse,
                   "tails_canaries_repeated_output": True, "public_api_passed": True}
            checks.append(row); print(json.dumps(row), flush=True)
        # A constant dyadic weight scale across groups reduces to the old
        # activation-only grouped formula; also check the backward-compatible API.
        channel_scale = torch.full((n,), .125, device="cuda", dtype=torch.bfloat16)
        repeated_scale = channel_scale[:, None].expand(n, k // 128).contiguous()
        old_reference = a_only_reference(qa, qw, sa, channel_scale).bfloat16()
        torch.testing.assert_close(wk.w4a4_grouped_gemm(ap, wp, sa, channel_scale), old_reference, rtol=0, atol=0)
        torch.testing.assert_close(wk.w4a4_grouped_gemm(ap, wp, sa, repeated_scale), old_reference, rtol=0, atol=0)
    report = {"complete": True, "gpu": gpu, "checks": checks, "constant_scale_and_backward_api": True,
              "kernel_sha256": sha256(Path(wk.__file__))}
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"complete": True, "checks": len(checks)}), flush=True)


if __name__ == "__main__":
    main()
