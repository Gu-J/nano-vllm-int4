"""Real nano TTFT and pure-decode ITL, serial prefill and a fixed decode batch."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import nano_bench_common as common


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['bf16', 'w4a4'], required=True)
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--input-tokens', type=int, default=4096)
    p.add_argument('--output-tokens', type=int, default=256)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    assert args.output_tokens >= 2
    gpu = common.setup(args.gpu, args.output_dir)
    import numpy as np
    import torch
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.scheduler import Scheduler
    from nanovllm.utils.context import reset_context

    print(json.dumps({'loading': args.backend, 'gpu': gpu}), flush=True)
    started = time.perf_counter()
    kwargs = dict(max_num_batched_tokens=args.input_tokens, max_num_seqs=args.batch,
                  max_model_len=args.input_tokens + args.output_tokens,
                  gpu_memory_utilization=0.9, enforce_eager=False)
    if args.backend == 'w4a4':
        kwargs['w4a4_checkpoint'] = str(common.CHECKPOINT)
    llm = LLM(common.MODEL, **kwargs)
    runner = llm.model_runner
    required_blocks = args.batch * ((args.input_tokens + args.output_tokens + 255) // 256)
    assert llm.config.num_kvcache_blocks >= required_blocks, 'Insufficient KV cache for fixed batch; refuse preemption'
    assert args.batch in runner.graphs, 'The measured decode batch must have its own CUDA graph'
    model_bytes = sum(t.numel() * t.element_size() for t in list(runner.model.parameters()) + list(runner.model.buffers()))
    report = {'complete': False, 'backend': args.backend, 'gpu': gpu, 'model': common.MODEL,
              'config': kwargs, 'batch': args.batch, 'input_tokens': args.input_tokens,
              'output_tokens': args.output_tokens, 'decode_steps_per_round': args.output_tokens - 1,
              'timing': 'Wall clock around actual LLM.step; includes scheduler, input transfer, model, LM head, sampler and token return; pretokenized requests; compilation/loading excluded.',
              'ttft': 'All requests arrive as one burst; first-token timestamps include serial-prefill queueing. Also report individual prefill service times.',
              'itl': 'Intervals of decode-only steps after all requests have their first token; excludes the prefill-to-decode waiting gap.',
              'prefix_cache': 'Scheduler/block manager recreated before each round; no cached prompt blocks.',
              'sampling': {'temperature': 0.6, 'ignore_eos': True},
              'graph_batches': runner.graph_bs, 'kv_blocks': llm.config.num_kvcache_blocks,
              'required_kv_blocks': required_blocks, 'model_tensor_bytes': model_bytes,
              'kv_cache_bytes': runner.kv_cache.numel() * runner.kv_cache.element_size(),
              'initialization_s': time.perf_counter() - started,
              'quantization': getattr(runner, 'quantization_info', None),
              'source_sha256': common.source_hashes(), 'torch': torch.__version__,
              'allow_bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
              'rounds': []}

    def stats(values):
        return {name: float(value) for name, value in [('mean', np.mean(values)), ('p50', np.median(values)),
                ('p95', np.quantile(values, .95)), ('min', np.min(values)), ('max', np.max(values))]}

    def run_round(index, output_tokens):
        common.exclusive(gpu)
        assert llm.is_finished()
        llm.scheduler = Scheduler(llm.config)
        inputs = common.prompts(args.batch, args.input_tokens, offset=max(index, 0) * 11)
        telemetry_before = subprocess.check_output(['nvidia-smi', '-i', gpu['uuid'],
            '--query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,memory.used',
            '--format=csv,noheader,nounits'], text=True).strip()
        torch.manual_seed(20261108 + max(index, 0))
        sampling = SamplingParams(temperature=.6, max_tokens=output_tokens, ignore_eos=True)
        torch.cuda.synchronize()
        arrival = time.perf_counter()
        for tokens in inputs:
            llm.add_request(tokens, sampling)
        seqs = list(llm.scheduler.waiting)
        first_times, prefill_ms, decode_ms, samples = {}, [], [], []
        while not llm.is_finished():
            t0 = time.perf_counter()
            _, work = llm.step()
            t1 = time.perf_counter()
            elapsed = (t1 - t0) * 1000
            if work > 0:
                assert work == args.input_tokens, ('Prefill must process exactly one full prompt', work)
                prefill_ms.append(elapsed)
                for seq in seqs:
                    if seq.num_completion_tokens and seq.seq_id not in first_times:
                        first_times[seq.seq_id] = (t1 - arrival) * 1000
            else:
                assert work == -args.batch and len(first_times) == args.batch, ('Decode batch changed', work)
                decode_ms.append(elapsed)
            samples.append({'phase': 'prefill' if work > 0 else 'decode', 'work_tokens': work,
                            'wall_ms': elapsed, 'elapsed_ms': (t1 - arrival) * 1000})
            if index == -1 and len(decode_ms) == 1 and 'warmup_graph_check' not in report:
                # Verify the actual long-context decode graph before timing it.
                ids, positions = runner.prepare_decode(list(llm.scheduler.running))
                graph_logits = runner.run_model(ids, positions, False).clone()
                runner.enforce_eager = True
                eager_logits = runner.run_model(ids, positions, False).clone()
                runner.enforce_eager = False
                reset_context()
                assert torch.isfinite(graph_logits).all()
                torch.testing.assert_close(graph_logits, eager_logits, rtol=0, atol=0)
                report['warmup_graph_check'] = {'batch': args.batch, 'context_tokens': args.input_tokens + 2,
                                                'full_logits_bit_equal': True}
        assert len(prefill_ms) == args.batch and len(decode_ms) == output_tokens - 1
        assert all(s.num_completion_tokens == output_tokens for s in seqs)
        common.exclusive(gpu)
        telemetry_after = subprocess.check_output(['nvidia-smi', '-i', gpu['uuid'],
            '--query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,memory.used',
            '--format=csv,noheader,nounits'], text=True).strip()
        return {'round': index, 'prompt_sha256': hashlib.sha256(json.dumps(inputs).encode()).hexdigest(),
                'telemetry_fields': 'temperature_C,sm_MHz,memory_MHz,power_W,memory_MiB',
                'telemetry_before': telemetry_before, 'telemetry_after': telemetry_after,
                'prefill_service_ms': stats(prefill_ms), 'ttft_ms': stats(list(first_times.values())),
                'pure_decode_itl_ms': stats(decode_ms), 'pure_decode_tokens_per_s': args.batch * 1000 / np.mean(decode_ms),
                'all_first_tokens_ms': max(first_times.values()), 'total_s': time.perf_counter() - arrival,
                'per_request_ttft_ms': [first_times[s.seq_id] for s in seqs], 'steps': samples,
                'output_token_ids': [s.completion_token_ids for s in seqs]}

    try:
        print(json.dumps({'warming': 'full 32-request prefill plus 8 decode steps'}), flush=True)
        warmup = run_round(-1, 9)
        report['warmup_s'] = warmup['total_s']
        torch.cuda.reset_peak_memory_stats()
        for index in range(args.rounds):
            item = run_round(index, args.output_tokens)
            report['rounds'].append(item)
            common.save(args.output_dir / 'results.json', report)
            print(json.dumps({k: item[k] for k in ('round', 'prefill_service_ms', 'ttft_ms', 'pure_decode_itl_ms', 'pure_decode_tokens_per_s')}), flush=True)
        report.update(complete=True, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
        common.save(args.output_dir / 'results.json', report)
    finally:
        llm.exit()


if __name__ == '__main__':
    main()
