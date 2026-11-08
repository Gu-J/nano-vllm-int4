"""RHT-aware GPTQ with static original-K G=128 weight scales."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

from gsm8k_eval import MODEL
from validate_rht import exclusive, select_gpu, sha256


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class CaptureDone(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--model", type=Path, default=Path(MODEL))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--damping", type=float, default=0.01)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--stop-after-layer", type=int, default=36)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent.parent
    for path in [args.data_dir, args.output_dir]:
        assert path.resolve().is_relative_to(repo) and path.resolve() != repo
    assert 1 <= args.stop_after_layer <= 36 and args.damping > 0 and args.block_size > 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(args.output_dir).free > 8 * 1024**3
    for name in ["HF_HOME", "HF_HUB_CACHE", "TORCH_HOME", "XDG_CACHE_HOME", "TRITON_CACHE_DIR"]:
        os.environ[name] = str((args.output_dir / "cache" / name.lower()).resolve())
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4")
    gpu = select_gpu(args.gpu)
    import torch
    import transformers
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM
    import rht_gptq_grouped as quant
    import rht_reference as ref
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    data_manifest = json.loads((args.data_dir / "manifest.json").read_text())
    token_path = args.data_dir / "calibration_tokens.safetensors"
    assert sha256(token_path) == data_manifest["tokens_sha256"]
    assert data_manifest["gsm8k_used"] is False
    tokens = load_file(str(token_path))["input_ids"]
    samples, length = tokens.shape
    assert [samples, length] == [data_manifest["samples"], data_manifest["sequence_length"]]
    config = {"model": str(args.model.resolve()), "model_config_sha256": sha256(args.model / "config.json"),
              "seed": args.seed, "layout_version": ref.LAYOUT_VERSION, "calibration": data_manifest,
              "hessian": "2 * mean(X_rotated_BF16.T @ X_rotated_BF16); A4 disabled during calibration",
              "calibration_order": "layer sequential; QKV -> O -> Gate/Up -> Down; propagate quantized weights",
              "weight_scales": "per-output-channel/per-original-K128-group absmax/7, BF16 BEFORE all GPTQ decisions",
              "weight_clipping_search": False, "weight_clipping": 1.0, "code_range": [-7, 7], "group_size": 128,
              "static_groups": True, "group_assignment": "original RHT K column // 128, including during act-order",
              "calibration_weight_backend": "exact sum of two BF16 parts, two cuBLAS FP32 GEMMs, BF16 output",
              "evaluation_activation_group_size": 128, "evaluation_activation_clipping": 0.9,
              "block_size": args.block_size, "damping_fraction": args.damping, "act_order": True,
              "runtime_column_permutation": False, "torch": torch.__version__,
              "transformers": transformers.__version__, "attention": "sdpa",
              "source_sha256": {name: sha256(Path(__file__).with_name(name)) for name in
                                  ["calibrate_rht_gptq_grouped.py", "rht_gptq_grouped.py", "rht_gptq.py",
                                   "rht_reference.py", "rht_fake_quant.py", "a4_group_quant.py", "prepare_gptq_data.py"]}}
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        assert manifest["config"] == config, "Resume configuration or source changed"
        if manifest["complete"]:
            print(json.dumps({"already_complete": True}), flush=True)
            return
    else:
        manifest = {"format": quant.FORMAT, "complete": False, "config": config, "gpu": gpu, "layers": {}}
        write_json(manifest_path, manifest)
    for name in ["weights", "reports"]:
        (args.output_dir / name).mkdir(exist_ok=True)
    started = time.monotonic()
    print(json.dumps({"loading": True, "gpu": gpu, "samples": samples, "length": length,
                      "completed_layers": len(manifest["layers"])}), flush=True)
    model = AutoModelForCausalLM.from_pretrained(str(args.model), local_files_only=True,
                                               torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval().cuda()
    model.config.use_cache = False
    hidden = model.config.hidden_size
    inps = torch.empty((samples, length, hidden), device="cuda", dtype=torch.bfloat16)
    outs = torch.empty_like(inps)
    assert len(model.model.layers) == 36 and all(layer.attention_type == "full_attention" for layer in model.model.layers)
    captured = {"index": 0, "kwargs": None}

    def capture(module, inputs, kwargs):
        inps[captured["index"]].copy_(inputs[0][0])
        captured["index"] += 1
        if captured["kwargs"] is None:
            captured["kwargs"] = kwargs
        assert kwargs["past_key_values"] is None and kwargs["use_cache"] is False
        raise CaptureDone()

    handle = model.model.layers[0].register_forward_pre_hook(capture, with_kwargs=True)
    try:
        for row in tokens:
            try:
                model.model(input_ids=row[None].cuda(), use_cache=False)
            except CaptureDone:
                pass
    finally:
        handle.remove()
    assert captured["index"] == samples
    layer_kwargs = captured["kwargs"]
    print(json.dumps({"captured_layer_inputs": list(inps.shape)}), flush=True)

    for layer_index, layer in enumerate(model.model.layers):
        exclusive(gpu)
        layer_started = time.monotonic()
        if str(layer_index) in manifest["layers"]:
            quant.load_layer(layer, layer_index, args.output_dir, manifest, bits=16)
            print(json.dumps({"resuming_layer": layer_index}), flush=True)
        else:
            packed_state, reports = {}, []
            for group, (parent_name, names, group_name) in enumerate(quant.GROUPS):
                exclusive(gpu)
                group_started = time.monotonic()
                parent = getattr(layer, parent_name)
                k = getattr(parent, names[0]).in_features
                signs = ref.make_signs(k, args.seed + layer_index * 17 + group, "cuda")
                collector = quant.InputHessian(k, signs)

                def collect(module, inputs):
                    collector.add(inputs[0])
                    raise CaptureDone()

                hook = getattr(parent, names[0]).register_forward_pre_hook(collect)
                try:
                    for sample in range(samples):
                        if sample % 32 == 0:
                            exclusive(gpu)
                        try:
                            layer(inps[sample:sample + 1], **layer_kwargs)
                        except CaptureDone:
                            pass
                finally:
                    hook.remove()
                assert collector.tokens == samples * length
                print(json.dumps({"layer": layer_index, "group": group_name, "hessian_tokens": collector.tokens,
                                  "collect_s": round(time.monotonic() - group_started, 2)}), flush=True)
                hessian = collector.finish()
                factor = quant.factor_hessian(hessian, args.damping, act_order=True)
                sampled_input = collector.sample
                del collector, hessian
                shared = quant.SharedActivation(signs, len(names), bits=16)
                packed_state[f"signs.{group_name}"] = signs.cpu().contiguous()
                for name in names:
                    old = getattr(parent, name)
                    assert old.bias is None
                    rotated_weight = ref.rht(old.weight, signs)
                    rtn_codes, scales = quant.quantize_groups(rotated_weight)
                    codes, details = quant.quantize_weight(rotated_weight, factor, scales, args.block_size)
                    reference = sampled_input @ rotated_weight.T
                    rtn_output = sampled_input @ quant.dequantize(rtn_codes, scales).T
                    gptq_output = sampled_input @ quant.dequantize(codes, scales).T
                    key = f"{parent_name}.{name}"
                    report = {"projection": key, "N": old.out_features, "K": old.in_features,
                              "hessian_tokens": samples * length, "dead_columns": factor["dead_columns"],
                              "actual_diagonal_damping": factor["damping"], "gptq": details,
                              "sampled_calibration_output_rtn": ref.error_stats(rtn_output, reference),
                              "sampled_calibration_output_gptq": ref.error_stats(gptq_output, reference)}
                    packed = ref.pack_int4(codes)
                    if layer_index == 0 and name == "q_proj":
                        torch.testing.assert_close(ref.unpack_int4(packed), codes, rtol=0, atol=0)
                    packed_state[key + ".packed"] = packed.cpu().contiguous()
                    packed_state[key + ".scale"] = scales.cpu().contiguous()
                    setattr(parent, name, quant.GPTQLinear(codes, scales, shared))
                    reports.append(report)
                    del old, rotated_weight, rtn_codes, codes, scales, reference, rtn_output, gptq_output, packed
                del factor, sampled_input, shared
                print(json.dumps({"layer": layer_index, "group": group_name,
                                  "group_s": round(time.monotonic() - group_started, 2)}), flush=True)
            weights_path = args.output_dir / "weights" / f"layer_{layer_index:02d}.safetensors"
            assert not weights_path.exists(), "Unindexed shard exists; inspect it before resuming"
            temporary = weights_path.with_suffix(".tmp")
            save_file(packed_state, str(temporary), metadata={"format": quant.FORMAT, "layer": str(layer_index)})
            temporary.replace(weights_path)
            del packed_state
            layer_report = {"layer": layer_index, "projections": reports,
                            "elapsed_s": time.monotonic() - layer_started}
            report_path = args.output_dir / "reports" / f"layer_{layer_index:02d}.json"
            write_json(report_path, layer_report)
            manifest["layers"][str(layer_index)] = {"file": str(weights_path.relative_to(args.output_dir)),
                                                     "bytes": weights_path.stat().st_size,
                                                     "sha256": sha256(weights_path),
                                                     "report": str(report_path.relative_to(args.output_dir)),
                                                     "report_sha256": sha256(report_path)}
            write_json(manifest_path, manifest)
            if layer_index == 0:
                before = layer(inps[0:1], **layer_kwargs).clone()
                quant.load_layer(layer, layer_index, args.output_dir, manifest, bits=16)
                after = layer(inps[0:1], **layer_kwargs)
                torch.testing.assert_close(after, before, rtol=0, atol=0)
                print(json.dumps({"first_layer_packed_reload_exact": True}), flush=True)
                del before, after
        for sample in range(samples):
            if sample % 32 == 0:
                exclusive(gpu)
            value = layer(inps[sample:sample + 1], **layer_kwargs)
            assert isinstance(value, torch.Tensor) and value.shape == (1, length, hidden)
            outs[sample].copy_(value[0])
        assert torch.isfinite(outs).all()
        inps, outs = outs, inps
        torch.cuda.empty_cache()
        progress = {"completed_layers": layer_index + 1, "total_layers": 36,
                    "layer_s": round(time.monotonic() - layer_started, 2),
                    "elapsed_s": round(time.monotonic() - started, 2),
                    "checkpoint_bytes": sum(v["bytes"] for v in manifest["layers"].values())}
        write_json(args.output_dir / "progress.json", progress)
        print(json.dumps(progress), flush=True)
        if layer_index + 1 >= args.stop_after_layer and layer_index + 1 < 36:
            print(json.dumps({"paused_after_requested_layer": layer_index + 1}), flush=True)
            return
    del inps, outs, layer_kwargs, captured
    smoke_tokens = tokens[0:1, :32].cuda()
    logits = model(smoke_tokens, use_cache=False, logits_to_keep=1).logits.cpu()
    smoke_path = args.output_dir / "reload_smoke.safetensors"
    save_file({"input_ids": smoke_tokens.cpu().contiguous(), "logits": logits.contiguous()}, str(smoke_path))
    manifest["reload_smoke_sha256"] = sha256(smoke_path)
    manifest["complete"] = True
    manifest["projections"] = 252
    manifest["peak_cuda_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
    write_json(manifest_path, manifest)
    print(json.dumps({"complete": True, "projections": 252, "packed_bytes": sum(v["bytes"] for v in manifest["layers"].values()),
                      "elapsed_s": time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    main()
