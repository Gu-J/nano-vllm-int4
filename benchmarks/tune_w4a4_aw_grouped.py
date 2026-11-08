"""Tune A128/W128 W4A4; frozen A128/W-PC, PTPC and cuBLAS are baselines."""
from benchmark_utils import source_path
import argparse
import csv
from dataclasses import asdict, replace
import gc
import hashlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import kernel_bench_common as common
from validate_rht import select_gpu, exclusive, sha256
from verify_w4a4_grouped import pack, reference as a_only_reference
from verify_w4a4_aw_grouped import reference


ROOT = Path("benchmarks/results/w4a4_aw_grouped")
A_ONLY_SHA = "8c35aa8fc2ef575918f1e5cd9e3d54570e9738639f218e2c3445a979229e8bfc"
A_ONLY_CONFIG_SHA = "9f1e235df719aa4ec792b69bc633d7fd3eb2019db6a0e3212d14b4ba03764727"


def load_a_only():
    path = Path(__file__).resolve().parent / "baselines/w4a4_a128_wpc/w4a4_grouped_kernel.py"
    assert sha256(path) == A_ONLY_SHA
    assert sha256(path.with_name("w4a4_grouped_configs.json")) == A_ONLY_CONFIG_SHA
    spec = importlib.util.spec_from_file_location("frozen_a128_wpc", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_data(m, n, k, seed):
    torch.manual_seed(seed + 31 * m + 17 * n + k)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = (torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / math.sqrt(k)).contiguous()
    x[0].zero_(); w[0].zero_()
    qa, sa = common.grouped_quantize(x, 128, .9)
    qap, sap = common.grouped_quantize(x, k, 1.)
    qwg, swg = common.grouped_quantize(w, 128, 1.)
    qwp, swp = common.grouped_quantize(w, k, 1.)
    sap, swp = sap.flatten(), swp.flatten()
    expected = reference(qa, qwg, sa, swg).bfloat16()
    a_expected = a_only_reference(qa, qwp, sa, swp).bfloat16()
    ptpc_expected = ((qap.float() @ qwp.float().T) * sap.float()[:, None] * swp.float()[None, :]).bfloat16()
    return {"M": m, "N": n, "K": k, "x": x, "w": w, "ag": pack(qa), "ap": pack(qap),
            "wg": pack(qwg), "wp": pack(qwp), "sa": sa, "sap": sap, "swg": swg, "swp": swp,
            "expected": expected, "a_expected": a_expected, "ptpc_expected": ptpc_expected,
            "out": torch.empty((m, n), device="cuda", dtype=torch.bfloat16)}


def call(data, config):
    return grouped.w4a4_grouped_gemm(data["ag"], data["wg"], data["sa"], data["swg"], data["out"], config)


def validate(data, config):
    data["out"].fill_(float("nan"))
    call(data, config)
    torch.testing.assert_close(data["out"], data["expected"], rtol=0, atol=0)
    return {"bf16_exact": True}


def initial_candidates(m, n, k):
    previous = grouped.GroupedConfig(**asdict(a_only.best_grouped_config(m, n, k)))
    candidates = [replace(previous, cache_scales=ac, cache_weight_scales=wc) for ac in [False, True] for wc in [False, True]]
    geometries = ([(16, 32, 64), (16, 64, 64), (16, 64, 128), (32, 64, 128), (16, 128, 128), (32, 128, 128)]
                  if m == 32 else [(64, 64, 128), (64, 128, 128), (128, 64, 128), (64, 256, 256), (128, 128, 256), (128, 128, 128)])
    return candidates + [grouped.GroupedConfig(bm, bn, 2, th, True, True) for bm, bn, th in geometries]


def compare(data, finalists, args, label):
    m, n, k = (data[key] for key in ["M", "N", "K"])
    ac = a_only.best_grouped_config(m, n, k)
    pc = ptpc.best_config(m, n, k)
    def a_call(cfg=ac):
        return a_only.w4a4_grouped_gemm(data["ag"], data["wp"], data["sa"], data["swp"], data["out"], cfg)
    def p_call():
        return ptpc.w4a4_gemm(data["ap"], data["wp"], data["sap"], data["swp"], data["out"], pc)
    a_call(); torch.testing.assert_close(data["out"], data["a_expected"], rtol=0, atol=0)
    p_call(); torch.testing.assert_close(data["out"], data["ptpc_expected"], rtol=0, atol=0)
    count = 128 if m == 32 else 16
    graphs = {}
    for i, cfg in enumerate(finalists):
        validate(data, cfg)
        graphs[f"aw_{i}"] = common.capture(lambda c=cfg: call(data, c), count)
    graphs["a128_wpc"] = common.capture(a_call, count)
    graphs["ptpc"] = common.capture(p_call, count)
    graphs["cublas_bf16"] = common.capture(lambda: torch.mm(data["x"], data["w"].T, out=data["out"]), count)
    exclusive(args.gpu_info)
    measured = common.measure(graphs, count, args.rounds, args.seed + m + n + k)
    index = min(range(len(finalists)), key=lambda i: measured[f"aw_{i}"]["median_us"])
    cfg = finalists[index]
    matched = a_only.GroupedConfig(cfg.bm, cfg.bn, cfg.stages, cfg.threads, cfg.cache_scales)
    a_call(matched); torch.testing.assert_close(data["out"], data["a_expected"], rtol=0, atol=0)
    graphs["a128_wpc_same_tile"] = common.capture(lambda: a_call(matched), count)
    chosen = {key: graphs[key] for key in [f"aw_{index}", "a128_wpc", "a128_wpc_same_tile", "ptpc", "cublas_bf16"]}
    before = common.telemetry(args.gpu_info)
    timing = common.measure(chosen, count, args.rounds, args.seed + 991 + m + n + k)
    after = common.telemetry(args.gpu_info)
    timing["a128_w128"] = timing.pop(f"aw_{index}")
    del chosen, graphs
    for _ in range(3):
        validate(data, cfg)
    source = grouped.compile_grouped_gemm(m, n, k, cfg, weight_grouped=True).get_kernel_source()
    assert "mma_sync<tl::DataType::kInt4, tl::DataType::kInt4, tl::DataType::kInt32" in source
    assert "atomicAdd" not in source and "atomic_add" not in source
    (args.output_dir / f"best_{label}_m{m}.cu").write_text(source)
    result = {"M": m, "N": n, "K": k, "name": label, "config": asdict(cfg), "a128_wpc_config": asdict(ac),
              "a128_wpc_same_tile_config": asdict(matched), "ptpc_config": asdict(pc), "timings": timing,
              "finalists": [asdict(c) for c in finalists], "finalist_timings": measured, "bf16_exact": True,
              "source_sha256": hashlib.sha256(source.encode()).hexdigest(), "telemetry_before": before, "telemetry_after": after}
    print(json.dumps({"shape_complete": result}), flush=True)
    return result


def run_shape(m, n, k, label, args, log, saved):
    data = make_data(m, n, k, args.seed)
    candidates, seen = [], set()
    def attempt(cfg, phase):
        if cfg in seen:
            return
        seen.add(cfg); exclusive(args.gpu_info)
        row = {"M": m, "N": n, "K": k, "name": label, "phase": phase, "config": asdict(cfg)}
        start = time.monotonic()
        try:
            validate(data, cfg)
            count = 32 if m == 32 else 4
            graph = common.capture(lambda: call(data, cfg), count)
            row["timing"] = common.measure({"aw": graph}, count, 5, args.seed)["aw"]
            del graph
            row["status"] = "pass"; row["bf16_exact"] = True; candidates.append(row)
        except Exception as error:
            row["status"] = "failed"; row["error"] = type(error).__name__ + ": " + str(error)[:5000]
            if any(s in str(error).lower() for s in ["illegal memory", "device-side assert", "misaligned address"]):
                raise
        row["elapsed_s"] = time.monotonic() - start
        log.write(json.dumps(row) + "\n"); log.flush()
        common.atomic_json(args.output_dir / "progress.json", {"status": "running", "pid": os.getpid(), "M": m,
                          "shape": label, "attempts": len(seen), "passing": len(candidates), "last": row})
        print(json.dumps({"candidate": row}), flush=True)
    if args.tune:
        for cfg in initial_candidates(m, n, k):
            attempt(cfg, "geometry_and_cache")
        assert candidates, "No valid candidate"
        seed = min(candidates, key=lambda r: r["timing"]["median_us"])
        cfg = grouped.GroupedConfig(**seed["config"])
        for stages in [1, 2, 3, 4]:
            for ac in [False, True]:
                for wc in [False, True]:
                    attempt(replace(cfg, stages=stages, cache_scales=ac, cache_weight_scales=wc), "pipeline_and_cache")
        finalists = [grouped.GroupedConfig(**r["config"]) for r in sorted(candidates, key=lambda r: r["timing"]["median_us"])[:3]]
    else:
        finalists = [grouped.GroupedConfig(**saved[f"{m},{n},{k}"])]
    result = compare(data, finalists, args, label)
    result.update(attempted=len(seen), passing=len(candidates))
    del data; gc.collect(); torch.cuda.empty_cache()
    return result


def save(report, output):
    common.atomic_json(output / "results.json", report)
    common.atomic_json(output / "w4a4_aw_grouped_configs.json", {f"{r['M']},{r['N']},{r['K']}": r["config"] for r in report["results"]})
    with (output / "results.csv").open("w", newline="") as handle:
        names = ["a128_w128", "a128_wpc", "a128_wpc_same_tile", "ptpc", "cublas_bf16"]
        writer = csv.DictWriter(handle, fieldnames=["M", "N", "K", "name"] + [s + "_us" for s in names] + ["aw_over_a_wpc", "aw_speedup_bf16"])
        writer.writeheader()
        for row in report["results"]:
            timings = row["timings"]; aw = timings["a128_w128"]["median_us"]
            writer.writerow({**{key: row[key] for key in ["M", "N", "K", "name"]},
                             **{s + "_us": timings[s]["median_us"] for s in names},
                             "aw_over_a_wpc": aw / timings["a128_wpc"]["median_us"], "aw_speedup_bf16": timings["cublas_bf16"]["median_us"] / aw})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--m", type=int, nargs="+", choices=[32, 4096], default=[32, 4096])
    p.add_argument("--shapes", nargs="+", choices=list(common.SHAPES), default=list(common.SHAPES))
    p.add_argument("--tune", action="store_true")
    p.add_argument("--configs", type=Path, default=source_path("w4a4_aw_grouped_configs.json"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=9)
    p.add_argument("--seed", type=int, default=20261108)
    args = p.parse_args()
    repo = Path(__file__).resolve().parent.parent
    assert args.output_dir.resolve().is_relative_to(repo)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(args.output_dir).free > 5 * 1024**3
    cache = ROOT / "cache" / f"gpu{args.gpu}"
    os.environ["TILELANG_CACHE_DIR"] = str(cache.resolve())
    os.environ["TILELANG_TMP_DIR"] = str((cache / "tmp").resolve())
    os.environ["OMP_NUM_THREADS"] = "1"
    args.gpu_info = select_gpu(args.gpu)
    global torch, grouped, a_only, ptpc
    import torch
    from nanovllm.kernels import w4a4_grouped_kernel as grouped
    common.torch = torch
    a_only, ptpc = load_a_only(), common.load_ptpc()
    torch.set_num_threads(1); torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    for name in list(logging.Logger.manager.loggerDict):
        if name.startswith("tilelang"):
            logging.getLogger(name).setLevel(logging.ERROR)
    config = {"gpu": args.gpu_info, "torch": torch.__version__, "tilelang": grouped.tilelang.__version__,
              "activation_group_size": 128, "weight_group_size": 128, "activation_clipping": .9, "weight_clipping": 1.,
              "scale_dtype": "BF16", "output_dtype": "BF16", "split_k": False, "atomic_add": False,
              "input_generation": "same seeded random BF16 X/W; symmetric INT4 codes [-7,7]; BF16 scales before quantization",
              "bf16_baseline": "torch.mm/cuBLAS FP32 reduction", "timing": "CUDA graph, preallocated output, shuffled rounds, warm buffers",
              "excluded_from_timing": ["RHT", "quantization", "packing", "compilation", "allocation"],
              "m_values": args.m, "shapes": args.shapes, "tuning": args.tune, "rounds": args.rounds, "seed": args.seed,
              "a128_wpc_sha256": A_ONLY_SHA, "a128_wpc_config_sha256": A_ONLY_CONFIG_SHA,
              "ptpc_sha256": common.PTPC_SHA, "ptpc_config_sha256": common.CONFIG_SHA,
              "source_sha256": {name: sha256(source_path(name)) for name in
                                ["w4a4_grouped_kernel.py", "tune_w4a4_aw_grouped.py", "verify_w4a4_aw_grouped.py", "kernel_bench_common.py"]}}
    assert not (args.output_dir / "results.json").exists(), "Use a fresh output directory"
    report = {"complete": False, "config": config, "results": []}
    saved = {} if args.tune else json.loads(args.configs.read_text())
    start = time.monotonic()
    common.atomic_json(args.output_dir / "progress.json", {"status": "running", "pid": os.getpid()})
    try:
        with (args.output_dir / "candidates.jsonl").open("a") as log:
            for m in args.m:
                for label in args.shapes:
                    exclusive(args.gpu_info)
                    n, k = common.SHAPES[label]
                    report["results"].append(run_shape(m, n, k, label, args, log, saved))
                    save(report, args.output_dir)
        report.update(complete=True, elapsed_s=time.monotonic()-start, peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
        save(report, args.output_dir)
        common.atomic_json(args.output_dir / "progress.json", {"status": "complete", "pid": os.getpid(), "shapes": len(report["results"]), "elapsed_s": report["elapsed_s"]})
    except BaseException:
        common.atomic_json(args.output_dir / "progress.json", {"status": "failed", "pid": os.getpid(), "error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
