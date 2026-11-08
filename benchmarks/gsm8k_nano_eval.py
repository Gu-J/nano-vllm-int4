"""Full GSM8K through the integrated native nano RHT/W4A4 model.

Only the generation controls are adapted: exact argmax and the saved two-EOS
stop rule. Model, attention, linear layers, kernels and configuration tables
are imported unchanged from nanovllm. No fake-quant model is instantiated.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

from gsm8k_eval import MODEL, prepare_data, normalize_number, extract_answers, parser_self_test, summarize
from gsm8k_gptq_grouped_eval import atomic_json, atomic_predictions
from validate_rht import select_gpu, exclusive, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, default=Path('benchmarks/results/gsm8k_rht/data'))
    p.add_argument('--checkpoint', type=Path, default=Path('benchmarks/results/rht_gptq_grouped/checkpoint'))
    p.add_argument('--baseline-dir', type=Path, default=Path('benchmarks/results/gsm8k_rht/bf16'))
    args = p.parse_args()
    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo))
    for path in (args.output_dir, args.data_dir, args.checkpoint):
        assert path.resolve().is_relative_to(repo) and path.resolve() != repo
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(args.output_dir).free > 2 * 1024**3
    parser_self_test()
    # Use the exact cached dataset and demonstrations, without updating them.
    assert (args.data_dir / 'gsm8k_test.jsonl').is_file() and (args.data_dir / 'gsm8k_cot.yaml').is_file()
    rows, prefix, data_manifest = prepare_data(args.data_dir)
    assert len(rows) == 1319
    baseline = json.loads((args.baseline_dir / 'config.json').read_text())
    expected_generation = json.loads((args.baseline_dir / 'effective_generation_config.json').read_text())
    baseline_rows = {r['id']: r for r in map(json.loads, (args.baseline_dir / 'predictions.jsonl').read_text().splitlines())}
    fake_config = json.loads((repo / 'benchmarks/results/gsm8k_gptq_grouped/worker_0/config.json').read_text())
    assert set(baseline_rows) == set(range(1319))
    for name in ('TORCH_HOME', 'XDG_CACHE_HOME', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'TILELANG_CACHE_DIR'):
        os.environ[name] = str(repo / 'benchmarks/results/nano_integration/cache' / name.lower())
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
    gpu = select_gpu(args.gpu)
    import torch
    import transformers
    from transformers import AutoTokenizer
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.scheduler import Scheduler
    from nanovllm.engine.sequence import SequenceStatus
    from nanovllm.layers.linear_w4a4 import W4A4Linear

    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    prompts = []
    for i, row in enumerate(rows):
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': prefix + f"\n\nQ: {row['question']}\nA:"}],
                                              tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        item = {'id': i, 'prompt': prompt, 'input_ids': ids, 'prompt_tokens': len(ids),
                'gold': normalize_number(row['answer'].rsplit('####', 1)[1])}
        assert item['gold'] == baseline_rows[i]['gold'] and item['prompt_tokens'] == baseline_rows[i]['prompt_tokens']
        prompts.append(item)
    assert prompts[0]['prompt'] == (args.baseline_dir / 'prompt_example.txt').read_text()
    prompt_hash = hashlib.sha256(json.dumps([r['prompt'] for r in prompts], ensure_ascii=False).encode()).hexdigest()
    assert prompt_hash == fake_config['all_prompts_sha256']
    ordered = sorted(prompts, key=lambda r: (r['prompt_tokens'], r['id']))
    batches = [ordered[i:i+32] for i in range(0, 1319, 32)]
    eos_ids = set(expected_generation['eos_token_id'])
    assert eos_ids == {151643, 151645} and not expected_generation['do_sample']
    assert expected_generation['max_new_tokens'] == 1024 and expected_generation['num_beams'] == 1
    assert max(r['prompt_tokens'] for r in prompts) + 1024 <= 4096
    runtime_sources = sorted((repo / 'nanovllm').rglob('*.py')) + sorted((repo / 'nanovllm/kernels').glob('*.json'))
    config = {**data_manifest, 'mode': 'nano_native_rht_wg128_ag128_c90', 'sample_count': 1319,
              'model_path': MODEL, 'model_config_sha256': sha256(Path(MODEL) / 'config.json'),
              'batch_size': 32, 'max_new_tokens': 1024, 'enable_thinking': False, 'do_sample': False,
              'num_beams': 1, 'seed': 2026, 'use_model_defaults': False, 'eos_token_ids': sorted(eos_ids),
              'strict_extractor': baseline['strict_extractor'], 'flexible_extractor': baseline['flexible_extractor'],
              'weight_group_size': 128, 'activation_group_size': 128, 'activation_clipping': .9,
              'scale_dtype': 'BF16', 'backend': 'integrated nano, TileLang RHT/quantize/pack + native INT4 GEMM',
              'attention': 'nano FlashAttention with BF16 paged KV cache', 'tensor_parallel_size': 1,
              'generation_adapter': 'argmax sampler, stop on either saved EOS; no model/kernel replacement',
              'batching': 'original prompt-length sorted groups of 32; native packed prefill and decode graph buckets; finished requests leave the active batch',
              'prefix_cache': 'fresh scheduler for each original batch; ordinary within-batch prefix cache',
              'engine_config': {'max_num_seqs': 32, 'max_num_batched_tokens': 4096, 'max_model_len': 4096,
                                'gpu_memory_utilization': .9, 'enforce_eager': False},
              'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_manifest_sha256': sha256(args.checkpoint / 'manifest.json'),
              'all_prompts_sha256': prompt_hash, 'ordered_batch_ids': [[r['id'] for r in batch] for batch in batches],
              'gpu': gpu, 'torch': torch.__version__, 'transformers': transformers.__version__,
              'runtime_source_sha256': {str(f.relative_to(repo)): sha256(f) for f in runtime_sources},
              'evaluation_source_sha256': {f.name: sha256(f) for f in [Path(__file__), Path(__file__).with_name('gsm8k_eval.py')]}}
    for key in ('protocol', 'test_sha256', 'fewshot_sha256', 'prefix_sha256', 'model_path', 'model_config_sha256',
                'batch_size', 'max_new_tokens', 'enable_thinking', 'do_sample', 'num_beams', 'seed',
                'use_model_defaults', 'strict_extractor', 'flexible_extractor', 'torch', 'transformers'):
        assert config[key] == baseline[key], key
    assert config['evaluation_source_sha256']['gsm8k_eval.py'] == baseline['source_sha256']['gsm8k_eval.py']
    assert config['checkpoint_manifest_sha256'] == fake_config['checkpoint_manifest_sha256']
    config_path, prediction_path = args.output_dir / 'config.json', args.output_dir / 'predictions.jsonl'
    records = []
    if config_path.exists():
        old = json.loads(config_path.read_text())
        for key in config.keys() - {'gpu'}:
            assert old.get(key) == config[key], f'Resume mismatch: {key}'
        if prediction_path.exists():
            records = [json.loads(line) for line in prediction_path.read_text().splitlines() if line]
    finished = {r['id'] for r in records}
    assert len(finished) == len(records) and finished <= set(range(1319))
    pending = []
    for index, batch in enumerate(batches):
        done = sum(r['id'] in finished for r in batch)
        assert done in (0, len(batch)), 'Partial saved batch'
        if not done: pending.append((index, batch))
    atomic_json(config_path, config)
    atomic_json(args.output_dir / 'effective_generation_config.json', expected_generation)
    (args.output_dir / 'prompt_example.txt').write_text(prompts[0]['prompt'])
    if not pending:
        print(json.dumps({'already_complete': 1319}), flush=True)
        return

    class GreedySampler(torch.nn.Module):
        def forward(self, logits, temperatures):
            return logits.argmax(dim=-1)

    class EvaluationScheduler(Scheduler):
        def postprocess(self, seqs, token_ids, is_prefill):
            lengths = [len(seq) for seq in seqs]
            super().postprocess(seqs, token_ids, is_prefill)
            for seq, token, before in zip(seqs, token_ids, lengths):
                if len(seq) == before + 1 and not seq.is_finished and token in eos_ids:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)

    greedy = GreedySampler()
    test_logits = torch.tensor([[1., 2., 2., -1.], [-9., -8., -7., -6.]], device='cuda', dtype=torch.bfloat16)
    assert greedy(test_logits, None).tolist() == [1, 3]
    started = time.monotonic()
    atomic_json(args.output_dir / 'progress.json', {'stage': 'loading', 'completed': len(records), 'total': 1319})
    print(json.dumps({'loading': 'native_nano_w4a4', 'gpu': gpu, 'samples': 1319, 'pending_batches': len(pending),
                      'prompt_tokens_range': [min(r['prompt_tokens'] for r in prompts), max(r['prompt_tokens'] for r in prompts)]}), flush=True)
    llm = LLM(MODEL, w4a4_checkpoint=str(args.checkpoint), **config['engine_config'])
    runner = llm.model_runner
    runner.sampler = greedy
    assert sum(isinstance(m, W4A4Linear) for m in runner.model.modules()) == 144
    assert runner.quantization_info['projections'] == 252 and 32 in runner.graphs
    assert runner.quantization_info['checkpoint_manifest_sha256'] == config['checkpoint_manifest_sha256']
    atomic_json(args.output_dir / 'conversion.json', runner.quantization_info)
    # Check the secondary EOS path, plus the ordinary non-EOS path, before evaluating.
    test_scheduler = EvaluationScheduler(llm.config)
    from nanovllm.engine.sequence import Sequence
    a, b = Sequence([1, 2]), Sequence([1, 3])
    test_scheduler.add(a); test_scheduler.add(b)
    seqs, is_prefill = test_scheduler.schedule()
    test_scheduler.postprocess(seqs, [151643, 5], is_prefill)
    assert a.is_finished and not b.is_finished and b.num_completion_tokens == 1
    atomic_json(args.output_dir / 'generation_adapter_check.json', {'greedy_ties_first_index': True, 'secondary_eos_stops': True, 'ordinary_token_continues': True})
    (args.output_dir / 'batch_audits').mkdir(exist_ok=True)
    previous_elapsed = 0
    if (args.output_dir / 'summary.json').exists():
        previous_elapsed = json.loads((args.output_dir / 'summary.json').read_text()).get('elapsed_s', 0) or 0
    try:
        with torch.inference_mode():
            for batch_index, batch in pending:
                exclusive(gpu)
                llm.scheduler = EvaluationScheduler(llm.config)
                sampling = SamplingParams(temperature=1., max_tokens=1024, ignore_eos=False)
                seq_ids = []
                for item in batch:
                    llm.add_request(item['input_ids'], sampling)
                    seq_ids.append(llm.scheduler.waiting[-1].seq_id)
                outputs, decode_steps, prefill_steps, histogram = {}, 0, 0, {}
                batch_started = last_heartbeat = time.monotonic()
                while not llm.is_finished():
                    output, work = llm.step()
                    outputs.update(output)
                    if work < 0:
                        decode_steps += 1
                        histogram[str(-work)] = histogram.get(str(-work), 0) + 1
                    else:
                        prefill_steps += 1
                    now = time.monotonic()
                    if now - last_heartbeat >= 30:
                        exclusive(gpu)
                        progress = {'stage': 'generating', 'completed': len(records), 'total': 1319,
                                    'global_batch_index': batch_index, 'batch_finished': len(outputs),
                                    'prefill_steps': prefill_steps, 'decode_steps': decode_steps,
                                    'batch_s': round(now - batch_started, 2)}
                        atomic_json(args.output_dir / 'progress.json', progress)
                        print(json.dumps(progress), flush=True)
                        last_heartbeat = now
                exclusive(gpu)
                assert set(outputs) == set(seq_ids)
                for item, seq_id in zip(batch, seq_ids):
                    values = outputs[seq_id]
                    assert 1 <= len(values) <= 1024
                    end = next((i for i, token in enumerate(values) if token in eos_ids), None)
                    sequence = values if end is None else values[:end]
                    if end is not None: assert end == len(values) - 1
                    completion = tokenizer.decode(sequence, skip_special_tokens=True)
                    prediction = extract_answers(completion)
                    records.append({'id': item['id'], 'gold': item['gold'], 'prediction': prediction,
                                    'correct': {name: value == item['gold'] for name, value in prediction.items()},
                                    'prompt_tokens': item['prompt_tokens'], 'generated_tokens': len(sequence),
                                    'finish_reason': 'length' if end is None else 'eos', 'stop_token_id': values[-1] if end is not None else None,
                                    'completion': completion, 'token_ids': sequence, 'global_batch_index': batch_index})
                atomic_predictions(prediction_path, records)
                audit = {'global_batch_index': batch_index, 'ids': [r['id'] for r in batch], 'samples': len(batch),
                         'prefill_steps': prefill_steps, 'decode_steps': decode_steps, 'decode_batch_histogram': histogram,
                         'batch_s': time.monotonic() - batch_started, 'native_linear_modules': 144,
                         'greedy': True, 'stop_token_ids': sorted(eos_ids)}
                atomic_json(args.output_dir / 'batch_audits' / f'batch_{batch_index:03d}.json', audit)
                summary = summarize(records, config, complete=len(records) == 1319,
                                    elapsed=previous_elapsed + time.monotonic() - started)
                summary['peak_cuda_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
                atomic_json(args.output_dir / 'summary.json', summary)
                progress = {'stage': 'complete' if summary['complete'] else 'running', 'completed': len(records), 'total': 1319,
                            'global_batch_index': batch_index, 'strict_accuracy': summary['strict']['accuracy'],
                            'flexible_accuracy': summary['flexible']['accuracy'], 'truncated': summary['truncated'],
                            'batch_s': round(audit['batch_s'], 2)}
                atomic_json(args.output_dir / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
        assert {r['id'] for r in records} == set(range(1319))
        for name, digest in config['runtime_source_sha256'].items():
            assert sha256(repo / name) == digest, f'Runtime changed during evaluation: {name}'
    finally:
        llm.exit()
    print(json.dumps({'complete': True, 'samples': 1319, 'strict': summary['strict'], 'flexible': summary['flexible']}), flush=True)


if __name__ == '__main__':
    main()
