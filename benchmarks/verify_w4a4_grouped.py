"""Independent references and correctness checks for grouped native INT4 GEMM."""
from benchmark_utils import source_path
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

from validate_rht import select_gpu, exclusive


def pack(q):
    return ((q[:, ::2] & 15) | ((q[:, 1::2] & 15) << 4)).contiguous()


def reference(qa, qw, sa, sw, dtype=None):
    import torch
    dtype = dtype or torch.float32
    m, k = qa.shape
    result = torch.zeros((m, qw.shape[0]), device=qa.device, dtype=dtype)
    for g in range(k // 128):
        dot = qa[:, g * 128:(g + 1) * 128].to(dtype) @ qw[:, g * 128:(g + 1) * 128].to(dtype).T
        result.add_(dot * sa[:, g].to(dtype)[:, None])
    return result * sw.to(dtype)[None, :]


def run_tests(gpu, output):
    import torch
    from nanovllm.kernels import w4a4_grouped_kernel as wk
    torch.manual_seed(20261109)
    checks = []
    shapes = [(17, 96, 512), (33, 136, 4096), (32, 128, 12288)]
    configs = [wk.GroupedConfig(16, 32, 2, 64, False), wk.GroupedConfig(32, 64, 2, 128, False),
               wk.GroupedConfig(32, 64, 2, 128, True)]
    for m, n, k in shapes:
        exclusive(gpu)
        qa = torch.randint(-8, 8, (m, k), device="cuda", dtype=torch.int8)
        qw = torch.randint(-8, 8, (n, k), device="cuda", dtype=torch.int8)
        qa[0].zero_()
        qw[0].zero_()
        sa = (torch.rand(m, k // 128, device="cuda") * 2).bfloat16()
        sw = (torch.rand(n, device="cuda") * .03).bfloat16()
        sa[1, ::2] = 0
        sw[1] = 0
        expected = reference(qa, qw, sa, sw)
        precise = reference(qa, qw, sa, sw, torch.float64)
        ap, wp = pack(qa), pack(qw)
        for cfg in configs:
            storage = torch.full((m * n + 128,), -1234, device="cuda", dtype=torch.bfloat16)
            sentinel = storage[:64].clone()
            result = storage[64:-64].reshape(m, n)
            for _ in range(3):
                result.fill_(float("nan"))
                wk.w4a4_grouped_gemm(ap, wp, sa, sw, result, cfg)
                torch.testing.assert_close(result, expected.bfloat16(), rtol=0, atol=0)
                torch.testing.assert_close(storage[:64], sentinel, rtol=0, atol=0)
                torch.testing.assert_close(storage[-64:], sentinel, rtol=0, atol=0)
            out32 = torch.empty((m, n), device="cuda", dtype=torch.float32)
            diagnostic = wk.compile_grouped_gemm(m, n, k, cfg, output_fp32=True)
            diagnostic(ap, wp, sa, sw, out32)
            torch.testing.assert_close(out32, expected, rtol=0, atol=0)
            rrmse = float((out32.double() - precise).square().mean().sqrt() /
                          precise.square().mean().sqrt().clamp_min(1e-12))
            assert rrmse < 2e-6
            source = wk.compile_grouped_gemm(m, n, k, cfg).get_kernel_source()
            assert "mma_sync<tl::DataType::kInt4, tl::DataType::kInt4, tl::DataType::kInt32" in source
            assert "atomicAdd" not in source and "atomic_add" not in source
            item = {"M": m, "N": n, "K": k, "config": asdict(cfg), "bf16_exact": True,
                    "fp32_exact_sequential_reference": True, "fp64_relative_rmse": rrmse,
                    "tails_and_canaries_passed": True, "repeated_overwrite_passed": True}
            checks.append(item)
            print(json.dumps({"check": item}), flush=True)
            (output / f"check_m{m}_n{n}_cache{int(cfg.cache_scales)}_bm{cfg.bm}.cu").write_text(source)
        # A constant dyadic scale must reduce to an ungrouped integer dot.
        sa.fill_(.125)
        got = wk.w4a4_grouped_gemm(ap, wp, sa, sw)
        expected_ptpc = ((qa.float() @ qw.float().T) * .125 * sw.float()[None, :]).bfloat16()
        torch.testing.assert_close(got, expected_ptpc, rtol=0, atol=0)
    return {"passed": True, "checks": checks, "constant_scale_reduces_to_ptpc": True,
            "kernel_sha256": hashlib.sha256(Path(wk.__file__).read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results/w4a4_grouped/check"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.output_dir.parent / "cache" / f"gpu{args.gpu}"
    os.environ["TILELANG_CACHE_DIR"] = str(cache.resolve())
    os.environ["TILELANG_TMP_DIR"] = str((cache / "tmp").resolve())
    gpu = select_gpu(args.gpu)
    import torch
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    report = run_tests(gpu, args.output_dir)
    (args.output_dir / "self_test.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"self_tests_complete": True, "checks": len(report["checks"])}), flush=True)


if __name__ == "__main__":
    main()
