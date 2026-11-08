"""Validate the RHT reference on unchanged Qwen3-8B BF16 activations.

Reads an existing local model. Never saves a model or activation tensors.
Results, input text, fixed signs and cache paths stay inside the output directory.
"""

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time
import urllib.request


DATA_URL = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt"
PROJECTIONS = {"q_proj": ("self_attn", "qkv"), "k_proj": ("self_attn", "qkv"),
               "v_proj": ("self_attn", "qkv"), "o_proj": ("self_attn", "o"),
               "gate_proj": ("mlp", "gate_up"), "up_proj": ("mlp", "gate_up"),
               "down_proj": ("mlp", "down")}
GROUP_IDS = {"qkv": 0, "o": 1, "gate_up": 2, "down": 3}


def exclusive(gpu):
    rows = csv.reader(subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True).splitlines())
    for row in rows:
        if row and row[0].strip() == gpu["uuid"] and int(row[1]) != os.getpid():
            raise RuntimeError(f"Another job appeared on target GPU: {row}")


def select_gpu(index):
    rows = csv.reader(subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True).splitlines())
    row = next([v.strip() for v in r] for r in rows if int(r[0]) == index)
    gpu = dict(zip(["index", "uuid", "name", "used_mib", "total_mib", "utilization"], row))
    exclusive(gpu)
    if int(gpu["used_mib"]) > 1024 or int(gpu["utilization"]) > 10:
        raise RuntimeError(f"GPU is not idle: {gpu}")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
    return gpu


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corpus(path):
    if not path.exists():
        request = urllib.request.Request(DATA_URL, headers={"User-Agent": "rht-reference-validation"})
        with urllib.request.urlopen(request, timeout=45) as response:
            data = response.read(4 * 1024**2 + 1)
        if not (100_000 < len(data) <= 4 * 1024**2):
            raise RuntimeError("Unexpected dataset size")
        data.decode("utf-8")
        path.write_bytes(data)
    return path.read_text(encoding="utf-8")


def activation_stats(x):
    q, scale = ref.quantize_rows(x)
    xf = x.float()
    ratio = xf.abs().amax(-1) / xf.square().mean(-1).sqrt().clamp_min(1e-12)
    return {"absmax_over_rms_median": ratio.median().item(),
            "absmax_over_rms_p95": torch.quantile(ratio, 0.95).item(),
            "absmax_over_rms_max": ratio.max().item(),
            "quant_error": ref.error_stats(q.float() * scale.float()[:, None], xf)}


def summarized(rows):
    summary = {}
    for name in PROJECTIONS:
        selected = [r for r in rows if r["projection"] == name]
        if not selected:
            continue
        before = [r["plain"]["w4a4"]["relative_rmse"] for r in selected]
        after = [r["rht"]["w4a4"]["relative_rmse"] for r in selected]
        ratios = [a / max(b, 1e-30) for a, b in zip(before, after)]
        summary[name] = {"layers": len(selected), "plain_rrmse_median": statistics.median(before),
                         "rht_rrmse_median": statistics.median(after),
                         "improvement_ratio_median": statistics.median(ratios),
                         "improved_layers": sum(a < b for a, b in zip(after, before)),
                         "worst_rht_layer": selected[max(range(len(after)), key=after.__getitem__)]["layer"],
                         "worst_rht_rrmse": max(after),
                         "min_improvement_ratio": min(ratios)}
    return summary


def save_report(report, output):
    report["summary"] = summarized(report["projections"])
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "projection_errors.csv").open("w", newline="") as f:
        fields = ["layer", "projection", "M", "N", "K", "plain_w4a4_rrmse", "rht_w4a4_rrmse",
                  "improvement_ratio", "rotation_identity_rrmse", "plain_w4a16_rrmse",
                  "rht_w4a16_rrmse", "plain_w16a4_rrmse", "rht_w16a4_rrmse"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in report["projections"]:
            p, h = r["plain"], r["rht"]
            writer.writerow(dict(layer=r["layer"], projection=r["projection"], M=r["M"], N=r["N"], K=r["K"],
                                 plain_w4a4_rrmse=p["w4a4"]["relative_rmse"], rht_w4a4_rrmse=h["w4a4"]["relative_rmse"],
                                 improvement_ratio=p["w4a4"]["relative_rmse"] / max(h["w4a4"]["relative_rmse"], 1e-30),
                                 rotation_identity_rrmse=r["rotation_identity"]["relative_rmse"],
                                 plain_w4a16_rrmse=p["w4a16"]["relative_rmse"], rht_w4a16_rrmse=h["w4a16"]["relative_rmse"],
                                 plain_w16a4_rrmse=p["w16a4"]["relative_rmse"], rht_w16a4_rrmse=h["w16a4"]["relative_rmse"]))


def capture_inputs(model, windows, sample_tokens, gpu):
    captured, handles = {}, []
    selected = torch.linspace(0, windows.shape[1] - 1, sample_tokens, device="cuda").round().long()
    for layer_id, layer in enumerate(model.model.layers):
        for group, module in [("qkv", layer.self_attn.q_proj), ("o", layer.self_attn.o_proj),
                              ("gate_up", layer.mlp.gate_proj), ("down", layer.mlp.down_proj)]:
            key = f"{layer_id}.{group}"
            captured[key] = []
            def hook(module, args, key=key):
                x = args[0].reshape(-1, args[0].shape[-1])
                captured[key].append(x.index_select(0, selected).detach().cpu())
            handles.append(module.register_forward_pre_hook(hook))
    try:
        for i, ids in enumerate(windows):
            exclusive(gpu)
            model.model(input_ids=ids.unsqueeze(0).cuda(), use_cache=False, return_dict=True)
            torch.cuda.synchronize()
            print(json.dumps({"capture_window": i + 1, "total": len(windows), "context": len(ids)}), flush=True)
    finally:
        for handle in handles:
            handle.remove()
    return {key: torch.cat(value) for key, value in captured.items()}


def compare_projection(x, w, signs):
    xf, wf = x.float(), w.float()
    xr, wr = ref.rht(xf, signs), ref.rht(wf, signs)
    target = xf @ wf.T
    identity = ref.error_stats(xr @ wr.T, target)
    if identity["relative_rmse"] > 1e-5:
        raise AssertionError(f"Rotation changed the unquantized operation: {identity}")
    result = {"rotation_identity": identity,
              "bf16_output_rounding": ref.error_stats(target.bfloat16(), target)}
    for name, xx, ww in [("plain", xf, wf), ("rht", xr, wr)]:
        qx, sx = ref.quantize_rows(xx)
        qw, sw = ref.quantize_rows(ww)
        result[name] = {
            "w4a4": ref.error_stats(ref.w4a4_reference(qx, sx, qw, sw), target),
            "w4a16": ref.error_stats(((xx @ qw.float().T) * sw.float()[None, :]).bfloat16(), target),
            "w16a4": ref.error_stats(((qx.float() @ ww.T) * sx.float()[:, None]).bfloat16(), target),
            "weight_quant_error": ref.error_stats(qw.float() * sw.float()[:, None], ww)}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--windows", type=int, default=4)
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--sample-tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--self-test-only", action="store_true")
    args = p.parse_args()
    if min(args.windows, args.context, args.sample_tokens) < 1 or args.sample_tokens > args.context:
        p.error("Invalid sample dimensions")
    repo = Path(__file__).resolve().parent.parent
    out = args.output_dir.resolve()
    if not out.is_relative_to(repo) or out == repo:
        p.error("Output must be a subdirectory of this development repository")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "report.json").exists():
        p.error("Report already exists; choose a new output directory")
    disk = shutil.disk_usage(out)
    if disk.free < 2 * 1024**3:
        raise RuntimeError("Less than 2 GiB free; refusing to create artifacts")
    for name, value in {"HF_HOME": out / "cache/hf", "HF_HUB_CACHE": out / "cache/hf/hub",
                        "TORCH_HOME": out / "cache/torch", "XDG_CACHE_HOME": out / "cache"}.items():
        os.environ[name] = str(value)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "4"
    gpu = select_gpu(args.gpu)
    global torch, ref
    import torch
    import rht_reference as ref
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    start = time.monotonic()
    report = {"stage": "RHT mathematical reference and local quantization errors", "complete": False,
              "model_path": str(args.model.resolve()), "gpu": gpu, "torch": torch.__version__,
              "seed": args.seed, "layout_version": ref.LAYOUT_VERSION,
              "source_sha256": {n: sha256(Path(__file__).with_name(n)) for n in ["rht_reference.py", "validate_rht.py"]},
              "disk_free_before_bytes": disk.free, "quantization": "absmax RTN [-7,7], BF16 per-token/per-channel scales rounded before codes; BF16 GEMM output",
              "scope": "Unchanged BF16 model captures; independent local projection errors, not accumulated W4A4 inference/PPL or performance",
              "projections": [], "activations": []}
    report["self_tests"] = ref.self_test("cuda")
    print(json.dumps({"self_tests_passed": True, "checks": report["self_tests"]}), flush=True)
    save_report(report, out)
    if args.self_test_only:
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers
    from safetensors.torch import save_file
    config = json.loads((args.model / "config.json").read_text())
    assert config["hidden_size"] == 4096 and config["intermediate_size"] == 12288 and config["num_hidden_layers"] == 36
    assert "quantization_config" not in config
    index = json.loads((args.model / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    report["model_shards"] = {n: (args.model / n).stat().st_size for n in shards}
    report["model_config_sha256"] = sha256(args.model / "config.json")
    report["transformers"] = transformers.__version__
    text_file = out / "wikitext2_valid.txt"
    text = corpus(text_file)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    ids = tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"]
    assert len(ids) >= args.context * args.windows
    offsets = torch.linspace(0, len(ids) - args.context, args.windows).round().long().tolist()
    windows = torch.tensor([ids[s:s + args.context] for s in offsets], dtype=torch.long)
    report["corpus"] = {"url": DATA_URL, "sha256": sha256(text_file), "text_bytes": text_file.stat().st_size,
                        "total_token_count": len(ids), "window_offsets": offsets, "context": args.context,
                        "window_count": args.windows, "tokens_per_window_retained": args.sample_tokens,
                        "retained_token_positions": torch.linspace(0, args.context - 1, args.sample_tokens).round().long().tolist()}
    print(json.dumps({"loading_existing_model": str(args.model), "download_model": False}), flush=True)
    exclusive(gpu)
    model = AutoModelForCausalLM.from_pretrained(str(args.model), torch_dtype=torch.bfloat16,
                                               local_files_only=True, attn_implementation="sdpa").eval().cuda()
    report["parameter_count"] = sum(v.numel() for v in model.parameters())
    original_versions = {n: p._version for n, p in model.named_parameters()}
    captured = capture_inputs(model, windows, args.sample_tokens, gpu)
    report["captured_activation_ram_bytes"] = sum(v.numel() * v.element_size() for v in captured.values())
    signs_cpu = {}
    for layer_id in range(len(model.model.layers)):
        for group in GROUP_IDS:
            k = captured[f"{layer_id}.{group}"].shape[-1]
            signs_cpu[f"layer_{layer_id}.{group}"] = ref.make_signs(k, args.seed + layer_id * 17 + GROUP_IDS[group])
    save_file(signs_cpu, str(out / "rotation_signs.safetensors"), metadata={"layout_version": ref.LAYOUT_VERSION})
    with (out / "projections.jsonl").open("w") as log:
        for layer_id, layer in enumerate(model.model.layers):
            exclusive(gpu)
            xs, signs = {}, {}
            for group in GROUP_IDS:
                x = captured.pop(f"{layer_id}.{group}").cuda()
                sign = signs_cpu[f"layer_{layer_id}.{group}"].cuda()
                xs[group], signs[group] = x, sign
                report["activations"].append({"layer": layer_id, "group": group, "M": x.shape[0], "K": x.shape[1],
                                               "plain": activation_stats(x), "rht": activation_stats(ref.rht(x, sign))})
            layer_rows = []
            for name, (parent, group) in PROJECTIONS.items():
                weight = getattr(getattr(layer, parent), name).weight
                row = {"layer": layer_id, "projection": name, "group": group, "M": xs[group].shape[0],
                       "N": weight.shape[0], "K": weight.shape[1],
                       **compare_projection(xs[group], weight, signs[group])}
                report["projections"].append(row)
                layer_rows.append(row)
                log.write(json.dumps(row) + "\n")
                log.flush()
            save_report(report, out)
            print(json.dumps({"layer_complete": layer_id, "of": len(model.model.layers),
                              "improved_projections": sum(r["rht"]["w4a4"]["relative_rmse"] < r["plain"]["w4a4"]["relative_rmse"] for r in layer_rows),
                              "down_plain_rrmse": layer_rows[-1]["plain"]["w4a4"]["relative_rmse"],
                              "down_rht_rrmse": layer_rows[-1]["rht"]["w4a4"]["relative_rmse"],
                              "elapsed_s": round(time.monotonic() - start)}), flush=True)
            del xs, signs
            gc.collect()
    assert all(p._version == original_versions[n] for n, p in model.named_parameters())
    report["model_parameters_unmodified"] = True
    report["complete"] = True
    report["elapsed_s"] = time.monotonic() - start
    report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["disk_free_after_bytes"] = shutil.disk_usage(out).free
    save_report(report, out)
    print(json.dumps({"complete": True, "projection_count": len(report["projections"]),
                      "summary": report["summary"], "peak_cuda_gib": report["peak_cuda_allocated_bytes"] / 1024**3,
                      "elapsed_s": report["elapsed_s"]}), flush=True)


if __name__ == "__main__":
    main()
