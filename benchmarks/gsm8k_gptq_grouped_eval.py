"""Full GSM8K with RHT-aware GPTQ W-G128, A-G128 clipping=0.9.

Workers partition the original complete batches by batch_index % shard_count.
Prompts, batching, greedy settings and answer extractors match the saved BF16 run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from gsm8k_eval import MODEL, prepare_data, normalize_number, extract_answers, parser_self_test, summarize
from validate_rht import select_gpu, exclusive, sha256


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def atomic_predictions(path, records):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--baseline-dir", type=Path, default=Path("benchmarks/results/gsm8k_rht/bf16"))
    p.add_argument("--model", type=Path, default=Path(MODEL))
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    args = p.parse_args()
    repo = Path(__file__).resolve().parent.parent
    for path in [args.output_dir, args.data_dir, args.checkpoint]:
        assert path.resolve().is_relative_to(repo) and path.resolve() != repo
    assert 0 <= args.shard_index < args.shard_count
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(args.output_dir).free > 2 * 1024**3
    parser_self_test()
    rows, prefix, data_manifest = prepare_data(args.data_dir)
    assert len(rows) == 1319
    checkpoint = json.loads((args.checkpoint / "manifest.json").read_text())
    assert checkpoint["complete"] and checkpoint["projections"] == 252
    assert checkpoint["config"]["model"] == str(args.model.resolve())
    assert checkpoint["config"]["model_config_sha256"] == sha256(args.model / "config.json")
    assert sha256(args.checkpoint / "reload_smoke.safetensors") == checkpoint["reload_smoke_sha256"]
    baseline = json.loads((args.baseline_dir / "config.json").read_text())
    expected_generation = json.loads((args.baseline_dir / "effective_generation_config.json").read_text())
    baseline_rows = {r["id"]: r for r in map(json.loads, (args.baseline_dir / "predictions.jsonl").read_text().splitlines())}
    assert set(baseline_rows) == set(range(1319))
    for name in ["HF_HOME", "HF_HUB_CACHE", "TORCH_HOME", "XDG_CACHE_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"]:
        os.environ[name] = str((args.output_dir / "cache" / name.lower()).resolve())
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4")
    gpu = select_gpu(args.gpu)
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    import rht_gptq_grouped as quant
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = []
    for i, row in enumerate(rows):
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prefix + f"\n\nQ: {row['question']}\nA:"}],
                                               tokenize=False, add_generation_prompt=True, enable_thinking=False)
        record = {"id": i, "prompt": prompt, "prompt_tokens": len(tokenizer.encode(prompt, add_special_tokens=False)),
                  "gold": normalize_number(row["answer"].rsplit("####", 1)[1])}
        assert record["gold"] == baseline_rows[i]["gold"]
        assert record["prompt_tokens"] == baseline_rows[i]["prompt_tokens"]
        prompts.append(record)
    assert prompts[0]["prompt"] == (args.baseline_dir / "prompt_example.txt").read_text()
    ordered = sorted(prompts, key=lambda r: (r["prompt_tokens"], r["id"]))
    full_batches = [ordered[i:i + args.batch_size] for i in range(0, len(ordered), args.batch_size)]
    selected_batches = [(i, batch) for i, batch in enumerate(full_batches) if i % args.shard_count == args.shard_index]
    selected_ids = [r["id"] for _, batch in selected_batches for r in batch]
    config = {**data_manifest, "mode": "gptq_wg128_ag128_c90", "sample_count": len(selected_ids),
              "weight_group_size": 128, "activation_group_size": 128, "activation_clipping": 0.9,
              "weight_clipping": 1.0, "scale_dtype": "BF16", "static_weight_groups": True,
              "backend": "exact normalized FP16 dequantization, FP32 cuBLAS accumulation, BF16 output",
              "model_path": str(args.model.resolve()), "model_config_sha256": sha256(args.model / "config.json"),
              "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens,
              "enable_thinking": False, "do_sample": False, "use_model_defaults": False,
              "num_beams": 1, "seed": 2026, "attention": "sdpa", "gpu": gpu,
              "torch": torch.__version__, "transformers": transformers.__version__,
              "strict_extractor": baseline["strict_extractor"], "flexible_extractor": baseline["flexible_extractor"],
              "checkpoint": str(args.checkpoint.resolve()), "checkpoint_manifest_sha256": sha256(args.checkpoint / "manifest.json"),
              "shard_index": args.shard_index, "shard_count": args.shard_count,
              "global_batch_indices": [i for i, _ in selected_batches], "selected_ids": selected_ids,
              "all_prompts_sha256": hashlib.sha256(json.dumps([r["prompt"] for r in prompts], ensure_ascii=False).encode()).hexdigest(),
              "source_sha256": {name: sha256(Path(__file__).with_name(name)) for name in
                                  ["gsm8k_gptq_grouped_eval.py", "gsm8k_eval.py", "rht_gptq_grouped.py",
                                   "rht_gptq.py", "rht_reference.py", "rht_fake_quant.py", "a4_group_quant.py"]}}
    protocol_keys = ["protocol", "test_sha256", "fewshot_sha256", "prefix_sha256", "model_config_sha256", "model_path",
                     "batch_size", "max_new_tokens", "enable_thinking", "do_sample", "use_model_defaults",
                     "num_beams", "seed", "attention", "torch", "transformers", "strict_extractor", "flexible_extractor"]
    for key in protocol_keys:
        assert config[key] == baseline[key], key
    assert config["source_sha256"]["gsm8k_eval.py"] == baseline["source_sha256"]["gsm8k_eval.py"]
    config_path = args.output_dir / "config.json"
    prediction_path = args.output_dir / "predictions.jsonl"
    records = []
    if config_path.exists():
        old = json.loads(config_path.read_text())
        for key in config.keys() - {"gpu"}:
            assert old.get(key) == config[key], f"Resume mismatch: {key}"
        if prediction_path.exists():
            records = [json.loads(line) for line in prediction_path.read_text().splitlines() if line]
    finished = {r["id"] for r in records}
    assert len(finished) == len(records) and finished <= set(selected_ids)
    pending = []
    for index, batch in selected_batches:
        done = sum(r["id"] in finished for r in batch)
        assert done in (0, len(batch)), "Incomplete saved batch"
        if not done:
            pending.append((index, batch))
    atomic_json(config_path, config)
    (args.output_dir / "prompt_example.txt").write_text(prompts[0]["prompt"])
    if not pending:
        atomic_json(args.output_dir / "summary.json", summarize(records, config, complete=True, elapsed=None))
        print(json.dumps({"already_complete": len(records)}), flush=True)
        return
    started = time.monotonic()
    print(json.dumps({"loading": "gptq_w4a4", "gpu": gpu, "samples": len(selected_ids), "global_batches": config["global_batch_indices"]}), flush=True)
    model = AutoModelForCausalLM.from_pretrained(str(args.model), local_files_only=True,
                                               torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval().cuda()
    conversion = quant.load_checkpoint(model, args.checkpoint, bits=16)
    reload_check = quant.check_reload(model, args.checkpoint)
    audit = quant.set_activation_bits(model, 4)
    conversion["activation_bits"] = 4
    conversion["calibration_activation_bits"] = 16
    conversion["weight_storage_exact"] = True
    model.eval()
    atomic_json(args.output_dir / "conversion.json", conversion)
    atomic_json(args.output_dir / "reload_validation.json", reload_check)
    print(json.dumps({"packed_reload_validation": reload_check, "projections": 252, "activation_bits": 4}), flush=True)
    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos)
    generation = GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1,
                                  pad_token_id=tokenizer.pad_token_id, eos_token_id=eos, use_cache=True)
    effective, _ = model._prepare_generation_config(generation, use_model_defaults=False, do_sample=False, num_beams=1)
    assert effective.to_dict() == expected_generation
    atomic_json(args.output_dir / "effective_generation_config.json", effective.to_dict())
    (args.output_dir / "batch_audits").mkdir(exist_ok=True)
    with torch.inference_mode():
        for global_index, batch in pending:
            exclusive(gpu)
            audit.zero_()
            encoded = tokenizer([r["prompt"] for r in batch], padding=True, add_special_tokens=False, return_tensors="pt").to("cuda")
            prompt_length = encoded["input_ids"].shape[1]
            assert prompt_length + args.max_new_tokens <= model.config.max_position_embeddings
            batch_started = time.monotonic()
            output = model.generate(**encoded, generation_config=generation, use_model_defaults=False,
                                    do_sample=False, num_beams=1, logits_to_keep=1)
            assert audit.item() == 0, "Inexact normalized FP16 activation storage; do not score this batch"
            atomic_json(args.output_dir / "activation_storage_audit.json", {
                "exact": True, "last_global_batch_index": global_index,
                "activation_group_size": 128, "activation_clipping": 0.9,
                "fp16_reconstruction_failures": 0})
            atomic_json(args.output_dir / "batch_audits" / f"batch_{global_index:03d}.json", {
                "global_batch_index": global_index, "samples": len(batch), "activation_group_size": 128,
                "activation_clipping": 0.9, "fp16_reconstruction_failures": 0})
            for item, values in zip(batch, output[:, prompt_length:].tolist()):
                end = next((i for i, token in enumerate(values) if token in eos_ids), None)
                sequence = values if end is None else values[:end]
                completion = tokenizer.decode(sequence, skip_special_tokens=True)
                prediction = extract_answers(completion)
                record = {"id": item["id"], "gold": item["gold"], "prediction": prediction,
                          "correct": {name: prediction[name] == item["gold"] for name in prediction},
                          "prompt_tokens": item["prompt_tokens"], "generated_tokens": len(sequence),
                          "finish_reason": "length" if end is None else "eos", "completion": completion,
                          "token_ids": sequence, "global_batch_index": global_index}
                records.append(record)
            atomic_predictions(prediction_path, records)
            summary = summarize(records, config, complete=len(records) == len(selected_ids), elapsed=time.monotonic() - started)
            summary["peak_cuda_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
            atomic_json(args.output_dir / "summary.json", summary)
            print(json.dumps({"shard": args.shard_index, "completed": len(records), "total": len(selected_ids),
                              "strict_accuracy": summary["strict"]["accuracy"], "flexible_accuracy": summary["flexible"]["accuracy"],
                              "truncated": summary["truncated"], "global_batch_index": global_index,
                              "batch_s": round(time.monotonic() - batch_started, 2)}), flush=True)
            del encoded, output
    print(json.dumps({"complete": True, "shard": args.shard_index, "total": len(records)}), flush=True)


if __name__ == "__main__":
    main()
