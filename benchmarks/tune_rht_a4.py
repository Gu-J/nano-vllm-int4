"""Tune only native RHT/A4 preparation; every candidate must match frozen codes/scales."""
from benchmark_utils import source_path
import argparse
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import time
import traceback

import rht_a4_bench_common as common
import bench_rht_a4 as checker
from validate_rht import select_gpu, exclusive, sha256


def configurations(k):
    geometry = ([kernel.Config(32, r) for r in [1, 2, 4, 8]] if k == 4096 else [])
    for threads in [64, 128, 256, 512]:
        for rows in ([1, 2, 4] if k == 4096 and threads < 512 else [1, 2]):
            geometry.append(kernel.Config(threads, rows))
    return geometry


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--rounds', type=int, default=9)
    args = p.parse_args()
    output = common.output_directory(args.output_dir)
    assert not (output / 'results.json').exists(), 'Use a fresh output directory'
    cache = common.REPO / f'benchmarks/results/rht_a4_native/cache/gpu{args.gpu}'
    os.environ['TILELANG_CACHE_DIR'] = str(cache)
    os.environ['TILELANG_TMP_DIR'] = str(cache / 'tmp')
    os.environ['OMP_NUM_THREADS'] = '1'
    manifest = common.checkpoint_manifest()
    gpu = select_gpu(args.gpu)
    global torch, kernel, timing
    import torch
    from nanovllm.kernels import rht_a4_tilelang as kernel
    import kernel_bench_common as timing
    from safetensors.torch import load_file
    timing.torch = torch
    torch.set_num_threads(1); torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    for name in list(logging.Logger.manager.loggerDict):
        if name.startswith('tilelang'):
            logging.getLogger(name).setLevel(logging.ERROR)
    fixtures = json.loads((common.ROOT / 'fixtures/manifest.json').read_text())
    fixture_file = common.ROOT / 'fixtures/activations.safetensors'
    assert sha256(fixture_file) == fixtures['data_sha256']
    fixture_data = load_file(str(fixture_file), device='cpu')
    report = {'complete': False, 'gpu': gpu, 'torch': torch.__version__, 'tilelang': kernel.tilelang.__version__,
              'contract': common.CONTRACT, 'checkpoint_manifest_sha256': sha256(common.CHECKPOINT / 'manifest.json'),
              'fixture_sha256': fixtures['data_sha256'], 'source_sha256': {
                  name: sha256(source_path(name)) for name in
                  ['rht_a4_tilelang.py', 'tune_rht_a4.py', 'rht_a4_bench_common.py', 'bench_rht_a4.py']},
              'rounds': args.rounds, 'timing': 'CUDA graphs, reused inputs/outputs, compilation excluded',
              'results': []}
    configs = {}
    start = time.monotonic()
    common.atomic_json(output / 'results.json', report)
    try:
        with (output / 'candidates.jsonl').open('a') as log:
            for k in [4096, 12288]:
                signs = common.checkpoint_signs(0, 'qkv' if k == 4096 else 'down', 'cuda')
                inputs, gold, outputs = {}, {}, {}
                for m in [32, 4096, 33]:
                    inputs[m] = common.make_input(m, k, 'cuda')
                    gold[m] = common.buffers(inputs[m])
                    common.reference_prepare(inputs[m], signs, out_packed=gold[m][0], out_scale=gold[m][1])
                    outputs[m] = common.buffers(inputs[m])
                real = []
                for case in fixtures['cases']:
                    if case['K'] == k:
                        real.append(tuple(fixture_data[f"{case['name']}.{key}"].cuda() for key in ['x', 'signs', 'packed', 'scale']))
                candidates = []
                for cfg in configurations(k):
                    exclusive(gpu)
                    begin = time.monotonic()
                    row = {'K': k, 'config': asdict(cfg)}
                    def adapter(x, s, *, out_packed, out_scale, config=cfg):
                        return kernel.rht_quant_pack(x, s, out_packed=out_packed, out_scale=out_scale, config=config)
                    try:
                        for m in [32, 4096, 33]:
                            checker.check_case(adapter, inputs[m], signs, gold[m])
                        for x, s, packed, scale in real:
                            checker.check_case(adapter, x, s, (packed, scale))
                        measurements = {}
                        for m in [32, 4096]:
                            def invoke(mm=m):
                                return adapter(inputs[mm], signs, out_packed=outputs[mm][0], out_scale=outputs[mm][1])
                            count = 128 if m == 32 else 16
                            graph = timing.capture(invoke, count)
                            measurements[str(m)] = timing.measure({'native': graph}, count, args.rounds, 20261108 + m + k)['native']
                            del graph
                        row.update(status='pass', codes_and_scales_exact=True, real_cases=len(real), timings=measurements)
                        candidates.append(row)
                    except Exception as error:
                        row.update(status='failed', error=type(error).__name__ + ': ' + str(error)[:5000])
                        if any(s in str(error).lower() for s in ['illegal memory', 'device-side assert', 'misaligned address']):
                            raise
                    row['elapsed_s'] = time.monotonic() - begin
                    log.write(json.dumps(row) + '\n'); log.flush()
                    common.atomic_json(output / 'progress.json', {'status': 'running', 'K': k, 'last': row, 'passing': len(candidates)})
                    print(json.dumps({'candidate': row}), flush=True)
                assert candidates, f'No correct candidates for K={k}'
                for m in [32, 4096]:
                    selected = sorted(candidates, key=lambda r: r['timings'][str(m)]['median_us'])[:3]
                    count = 32 if m == 32 else 4
                    graphs = {}
                    for i, entry in enumerate(selected):
                        cfg = kernel.Config(**entry['config'])
                        def invoke(config=cfg):
                            return kernel.rht_quant_pack(inputs[m], signs, out_packed=outputs[m][0], out_scale=outputs[m][1], config=config)
                        graphs[f'native_{i}'] = timing.capture(invoke, count)
                    graphs['pytorch_reference'] = timing.capture(lambda: common.reference_prepare(
                        inputs[m], signs, out_packed=outputs[m][0], out_scale=outputs[m][1]), count)
                    exclusive(gpu)
                    before = timing.telemetry(gpu)
                    final_times = timing.measure(graphs, count, args.rounds, 20261108 + m + k + 11)
                    winner = min(range(len(selected)), key=lambda i: final_times[f'native_{i}']['median_us'])
                    cfg = kernel.Config(**selected[winner]['config'])
                    configs[f'{m},{k}'] = asdict(cfg)
                    source = kernel.compile_kernel(k, cfg).get_kernel_source()
                    assert 'atomicAdd' not in source and 'atomic_add' not in source
                    (output / f'best_m{m}_k{k}.cu').write_text(source)
                    result = {'M': m, 'K': k, 'config': asdict(cfg),
                              'native': final_times[f'native_{winner}'], 'pytorch_reference': final_times['pytorch_reference'],
                              'finalist_configs': [entry['config'] for entry in selected], 'finalist_timings': final_times,
                              'replays_per_sample': count, 'codes_and_scales_exact': True,
                              'source_sha256': __import__('hashlib').sha256(source.encode()).hexdigest(),
                              'telemetry_before': before, 'telemetry_after': timing.telemetry(gpu),
                              'passing_configs': len(candidates), 'attempted_configs': len(configurations(k))}
                    report['results'].append(result)
                    common.atomic_json(output / 'results.json', report)
                    common.atomic_json(output / 'rht_a4_configs.json', configs)
                    print(json.dumps({'shape_complete': result}), flush=True)
                    del graphs
                del inputs, gold, outputs, real
                torch.cuda.empty_cache()
        report.update(complete=True, elapsed_s=time.monotonic() - start,
                      peak_cuda_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3)
        common.atomic_json(output / 'results.json', report)
        common.atomic_json(output / 'progress.json', {'status': 'complete', 'elapsed_s': report['elapsed_s']})
    except BaseException:
        common.atomic_json(output / 'progress.json', {'status': 'failed', 'error': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
