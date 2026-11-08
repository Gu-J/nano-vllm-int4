"""Paired GSM8K accuracy evaluation: BF16 versus uniform RHT W4A4 fake quant.

8-shot CoT demonstrations from lm-evaluation-harness, native Qwen chat template,
thinking disabled, greedy decoding. This is not a claim to reproduce Qwen's
published evaluation protocol. No model checkpoint is written.
"""
import argparse
import copy
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
import urllib.request

from validate_rht import select_gpu, exclusive, sha256

TEST_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
SHOT_URL = "https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/lm_eval/tasks/gsm8k/gsm8k-cot.yaml"
MODEL = "/home/huggingface/Qwen3-8B-b968826d9c46dd6066d109eabc6255188de91218"
PROTOCOL = "gsm8k_8shot_cot_qwen_chat_nonthinking_greedy_v1"
INSTRUCTION = 'Solve the math word problem. Show your reasoning, and end with "The answer is <number>."\n\n'
NUMBER = r"[-+]?\d[\d,]*(?:\.\d+)?"


def normalize_number(text):
    try:
        value = Decimal(text.replace(",", "").replace("$", "").strip().rstrip("."))
        return str(value.normalize()) if value.is_finite() else None
    except (InvalidOperation, AttributeError):
        return None


def extract_answers(text):
    strict = re.search(r"The answer is\s*:?[ \t]*\$?[ \t]*(" + NUMBER + r")", text, flags=re.I)
    numbers = re.findall(NUMBER, text)
    return {"strict": normalize_number(strict.group(1)) if strict else None,
            "flexible": normalize_number(numbers[-1]) if numbers else None}


def parser_self_test():
    for text, expected in [("The answer is 18.", "18"), ("the answer is $1,234.00.", "1234"),
                           ("The answer is -0.25.", "-0.25"), ("Work: 9 * 2 = 18.", None)]:
        value = extract_answers(text)["strict"]
        assert value == (normalize_number(expected) if expected else None), (text, value)
    assert extract_answers("First 12, then 15.")["flexible"] == normalize_number("15")
    assert extract_answers("No numeric answer.") == {"strict": None, "flexible": None}


def download(url, path):
    if path.exists():
        return
    request = urllib.request.Request(url, headers={"User-Agent": "nano-vllm-gsm8k-evaluation"})
    with urllib.request.urlopen(request, timeout=45) as response:
        data = response.read(5 * 1024**2 + 1)
    if not (100 < len(data) <= 5 * 1024**2):
        raise RuntimeError("Unexpected data download size")
    data.decode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def prepare_data(directory):
    import yaml
    directory.mkdir(parents=True, exist_ok=True)
    test, shots = directory / "gsm8k_test.jsonl", directory / "gsm8k_cot.yaml"
    download(TEST_URL, test)
    download(SHOT_URL, shots)
    rows = [json.loads(line) for line in test.read_text().splitlines() if line]
    examples = yaml.safe_load(shots.read_text())["fewshot_config"]["samples"]
    assert len(rows) == 1319 and len(examples) == 8
    assert not ({r["question"] for r in rows} & {r["question"] for r in examples})
    prefix = INSTRUCTION + "\n\n".join(f"Q: {r['question']}\nA: {r['target']}" for r in examples)
    manifest = {"protocol": PROTOCOL, "test_url": TEST_URL, "test_sha256": sha256(test),
                "fewshot_url": SHOT_URL, "fewshot_sha256": sha256(shots), "test_count": len(rows),
                "fewshot_count": len(examples), "prefix_sha256": hashlib.sha256(prefix.encode()).hexdigest()}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return rows, prefix, manifest


def wilson(correct, total):
    if total == 0:
        return [0.0, 1.0]
    z, probability = 1.959963984540054, correct / total
    denominator = 1 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    delta = z * math.sqrt(probability * (1 - probability) / total + z * z / (4 * total * total)) / denominator
    return [center - delta, center + delta]


def summarize(records, config, complete=False, elapsed=0):
    n = len(records)
    result = {"complete": complete, "mode": config["mode"], "total": n, "expected_total": config["sample_count"],
              "elapsed_s": elapsed, "truncated": sum(r["finish_reason"] == "length" for r in records),
              "strict_parse_failures": sum(r["prediction"]["strict"] is None for r in records),
              "generated_tokens": sum(r["generated_tokens"] for r in records), "protocol": config}
    for name in ["strict", "flexible"]:
        count = sum(r["correct"][name] for r in records)
        result[name] = {"correct": count, "accuracy": count / n if n else None, "wilson_95": wilson(count, n)}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["bf16", "rht_w4a4"], default="bf16")
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--model", type=Path, default=Path(MODEL))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    repo = Path(__file__).resolve().parent.parent
    for path in [args.output_dir, args.data_dir]:
        if not path.resolve().is_relative_to(repo) or path.resolve() == repo:
            p.error("All new artifacts must be inside the development repository")
        path.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output_dir).free < 2 * 1024**3:
        raise RuntimeError("Less than 2 GiB free")
    parser_self_test()
    rows, prefix, data_manifest = prepare_data(args.data_dir)
    if args.prepare_only:
        print(json.dumps(data_manifest), flush=True)
        return
    if args.limit:
        rows = rows[:args.limit]
    if args.batch_size < 1 or args.max_new_tokens < 1:
        p.error("Positive batch size and generation cap required")
    for name, value in {"HF_HOME": "cache/hf", "HF_HUB_CACHE": "cache/hf/hub", "TORCH_HOME": "cache/torch",
                        "XDG_CACHE_HOME": "cache", "TORCHINDUCTOR_CACHE_DIR": "cache/inductor",
                        "TRITON_CACHE_DIR": "cache/triton"}.items():
        os.environ[name] = str((args.output_dir / value).resolve())
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4")
    gpu = select_gpu(args.gpu)
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = []
    for i, row in enumerate(rows):
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prefix + f"\n\nQ: {row['question']}\nA:"}],
                                               tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompts.append({"id": i, "prompt": prompt, "prompt_tokens": len(tokenizer.encode(prompt, add_special_tokens=False)),
                        "gold": normalize_number(row["answer"].rsplit("####", 1)[1])})
    config = {**data_manifest, "mode": args.mode, "sample_count": len(rows), "model_path": str(args.model.resolve()),
              "model_config_sha256": sha256(args.model / "config.json"), "batch_size": args.batch_size,
              "max_new_tokens": args.max_new_tokens, "enable_thinking": False, "do_sample": False,
              "use_model_defaults": False,
              "num_beams": 1, "seed": 2026, "attention": "sdpa", "gpu": gpu,
              "torch": torch.__version__, "transformers": transformers.__version__,
              "strict_extractor": "first case-insensitive 'The answer is' numeric value; commas/currency normalized",
              "flexible_extractor": "last numeric value in the generated completion",
              "source_sha256": {"gsm8k_eval.py": sha256(Path(__file__)),
                                  "rht_reference.py": sha256(Path(__file__).with_name("rht_reference.py"))}}
    if args.mode == "rht_w4a4":
        config["source_sha256"]["rht_fake_quant.py"] = sha256(Path(__file__).with_name("rht_fake_quant.py"))
    manifest_path = args.output_dir / "config.json"
    records, finished = [], set()
    prediction_file = args.output_dir / "predictions.jsonl"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        # GPU transient memory is irrelevant to deterministic resume.
        for name in config.keys() - {"gpu"}:
            if old.get(name) != config[name]:
                raise RuntimeError(f"Existing run has a different setting: {name}")
        if prediction_file.exists():
            records = [json.loads(line) for line in prediction_file.read_text().splitlines() if line]
            finished = {r["id"] for r in records}
            assert len(finished) == len(records)
    manifest_path.write_text(json.dumps(config, indent=2) + "\n")
    (args.output_dir / "prompt_example.txt").write_text(prompts[0]["prompt"])
    pending = sorted([r for r in prompts if r["id"] not in finished], key=lambda r: (r["prompt_tokens"], r["id"]))
    if not pending:
        print(json.dumps({"already_complete": len(records)}), flush=True)
        return
    start = time.monotonic()
    print(json.dumps({"loading": args.mode, "gpu": gpu, "samples": len(rows), "pending": len(pending),
                      "prompt_tokens_range": [min(r['prompt_tokens'] for r in prompts), max(r['prompt_tokens'] for r in prompts)]}), flush=True)
    model = AutoModelForCausalLM.from_pretrained(str(args.model), local_files_only=True,
                                               torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval().cuda()
    if args.mode == "rht_w4a4":
        from rht_fake_quant import convert_model, backend_self_test
        validation = backend_self_test(args.batch_size)
        (args.output_dir / "fake_quant_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
        print(json.dumps({"fake_quant_backend_pass": validation}), flush=True)
        conversion = convert_model(model, seed=2026)
        (args.output_dir / "conversion.json").write_text(json.dumps(conversion, indent=2) + "\n")
        print(json.dumps({"converted": conversion}), flush=True)
    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos)
    generation = GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1,
                                  pad_token_id=tokenizer.pad_token_id, eos_token_id=eos, use_cache=True)
    effective, _ = model._prepare_generation_config(generation, use_model_defaults=False, do_sample=False, num_beams=1)
    assert effective.do_sample is False and effective.num_beams == 1
    (args.output_dir / "effective_generation_config.json").write_text(json.dumps(effective.to_dict(), indent=2) + "\n")
    print(json.dumps({"effective_decoding": str(effective.get_generation_mode()), "do_sample": effective.do_sample}), flush=True)
    with prediction_file.open("a") as log, torch.inference_mode():
        for begin in range(0, len(pending), args.batch_size):
            exclusive(gpu)
            batch = pending[begin:begin + args.batch_size]
            encoded = tokenizer([r["prompt"] for r in batch], padding=True, add_special_tokens=False, return_tensors="pt").to("cuda")
            prompt_length = encoded["input_ids"].shape[1]
            if prompt_length + args.max_new_tokens > model.config.max_position_embeddings:
                raise RuntimeError("Requested generation exceeds model context")
            batch_start = time.monotonic()
            output = model.generate(**encoded, generation_config=generation, use_model_defaults=False,
                                    do_sample=False, num_beams=1, logits_to_keep=1)
            for item, tokens in zip(batch, output[:, prompt_length:].tolist()):
                end = next((i for i, token in enumerate(tokens) if token in eos_ids), None)
                sequence = tokens if end is None else tokens[:end]
                completion = tokenizer.decode(sequence, skip_special_tokens=True)
                prediction = extract_answers(completion)
                record = {"id": item["id"], "gold": item["gold"], "prediction": prediction,
                          "correct": {name: prediction[name] == item["gold"] for name in prediction},
                          "prompt_tokens": item["prompt_tokens"], "generated_tokens": len(sequence),
                          "finish_reason": "length" if end is None else "eos", "completion": completion,
                          "token_ids": sequence}
                records.append(record)
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.flush()
            os.fsync(log.fileno())
            summary = summarize(records, config, complete=len(records) == len(rows), elapsed=time.monotonic() - start)
            summary["peak_cuda_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
            (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps({"mode": args.mode, "completed": len(records), "total": len(rows),
                              "strict_accuracy": summary["strict"]["accuracy"], "flexible_accuracy": summary["flexible"]["accuracy"],
                              "truncated": summary["truncated"], "batch_s": round(time.monotonic() - batch_start, 2),
                              "elapsed_s": round(time.monotonic() - start)}), flush=True)
            del output, encoded
    print(json.dumps({"complete": True, "mode": args.mode, "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
