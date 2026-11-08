"""Validate an RHT/A4 adapter or measure preparation + native grouped GEMM.

`reference` exercises the harness only. A requested module:function is never
silently replaced by the reference. This file does not implement a native RHT.
"""
from benchmark_utils import source_path
import argparse
import json
import os
from pathlib import Path
import traceback

import rht_a4_bench_common as common
from validate_rht import select_gpu, exclusive, sha256


def oracle_checks():
    import math
    import torch
    import rht_reference as ref
    import a4_group_quant as a4
    # Analytic impulse responses fix ordering and H12 orientation independently.
    for k, index in [(4096, 5), (12288, 1024)]:
        x = torch.zeros(1, k, dtype=torch.float64); x[0, index] = 1
        signs = torch.ones(k, dtype=torch.int8)
        actual = ref.rht(x, signs)
        if k == 4096:
            expected = torch.tensor([(-1 if (j & index).bit_count() % 2 else 1) / math.sqrt(k)
                                     for j in range(k)], dtype=torch.float64)
        else:
            # Paley row 1: diagonal -1, quadratic residues +1, others -1.
            residues = {i * i % 11 for i in range(1, 11)}
            row = [1] + [(-1 if j == 0 else 1 if j in residues else -1) for j in range(11)]
            expected = torch.tensor(row, dtype=torch.float64).repeat_interleave(1024) / math.sqrt(k)
        torch.testing.assert_close(actual[0], expected, rtol=1e-12, atol=1e-12)
    values = torch.zeros(1, 128)
    values[0, :10] = torch.tensor([-7.5, -6.5, -4.5, -2.5, -.5, .5, 2.5, 4.5, 6.5, 7.5])
    values[0, -1] = 7 / .9
    codes, scales = a4.quantize_rotated(values, 128, .9)
    assert scales.item() == 1
    assert codes.flatten()[:10].tolist() == [-7, -6, -4, -2, 0, 0, 2, 4, 6, 7]
    raw = torch.tensor(.9 / 7); rounded = raw.bfloat16().float()
    values.zero_(); values[0, 0] = 1; values[0, 1] = 2.5 * (raw + rounded) / 2
    codes, scales = a4.quantize_rotated(values, 128, .9)
    assert codes.flatten()[1].item() == 2 and torch.round(values[0, 1] / raw).item() == 3
    for magnitude, expected_scale in [(0., 1.), (1e-40, torch.finfo(torch.bfloat16).tiny)]:
        codes, scales = a4.quantize_rotated(torch.full((1, 128), magnitude), 128, .9)
        assert scales.item() == expected_scale and codes.count_nonzero().item() == 0
    codes = torch.arange(-8, 8, dtype=torch.int8).reshape(1, 16)
    assert (ref.pack_int4(codes).to(torch.int16) & 255).flatten().tolist() == [0x98, 0xba, 0xdc, 0xfe, 0x10, 0x32, 0x54, 0x76]
    return {'analytic_rht_orientation': True, 'ties_to_even_and_clamp': True,
            'bf16_scale_before_codes': True, 'zero_and_tiny_groups': True, 'known_packed_bytes': True}


def compare(actual_packed, actual_scale, gold_packed, gold_scale):
    import torch
    import rht_reference as ref
    mismatched_codes = int((ref.unpack_int4(actual_packed) != ref.unpack_int4(gold_packed)).sum().item())
    mismatched_scales = int((actual_scale != gold_scale).sum().item())
    assert mismatched_codes == mismatched_scales == 0, (
        f'Quantizer parity failed: {mismatched_codes} codes, {mismatched_scales} BF16 scales differ')


def check_case(function, x, signs, gold=None):
    import torch
    import rht_reference as ref
    assert x.dtype == torch.bfloat16 and signs.dtype == torch.int8
    assert x.is_contiguous() and signs.is_contiguous()
    m, k = x.shape
    x_before, signs_before = x.clone(), signs.clone()
    if gold is None:
        _, codes, scale = common.reference_parts(x, signs)
        gold = ref.pack_int4(codes), scale
    packed_store = torch.full((m * k // 2 + 128,), -69, dtype=torch.int8, device=x.device)
    scale_store = torch.full((m * k // 128 + 128,), -17, dtype=torch.bfloat16, device=x.device)
    packed = packed_store[64:-64].view(m, k // 2)
    scale = scale_store[64:-64].view(m, k // 128)
    for _ in range(2):
        packed.fill_(85); scale.fill_(float('nan'))
        returned = function(x, signs, out_packed=packed, out_scale=scale)
        assert isinstance(returned, tuple) and len(returned) == 2
        assert returned[0].data_ptr() == packed.data_ptr() and returned[1].data_ptr() == scale.data_ptr()
        compare(packed, scale, *gold)
        assert torch.isfinite(scale).all() and (scale > 0).all()
        assert (packed_store[:64] == -69).all() and (packed_store[-64:] == -69).all()
        assert (scale_store[:64] == -17).all() and (scale_store[-64:] == -17).all()
        assert torch.equal(x, x_before) and torch.equal(signs, signs_before)
    return {'M': m, 'K': k, 'codes_exact': True, 'bf16_scales_exact': True,
            'preallocated_outputs': True, 'canaries_and_inputs_unchanged': True}


def dynamic_checks(function, k):
    import torch
    import rht_reference as ref
    x = common.make_input(32, k, 'cuda')
    signs = common.checkpoint_signs(0, 'qkv' if k == 4096 else 'down', 'cuda')
    packed, scale = common.buffers(x)
    def invoke():
        return function(x, signs, out_packed=packed, out_scale=scale)
    for _ in range(3):
        invoke()
    stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        invoke()
    torch.cuda.current_stream().wait_stream(stream)
    _, codes, scales = common.reference_parts(x, signs)
    compare(packed, scale, ref.pack_int4(codes), scales)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke()
    for offset in [.125, -.25]:
        x.add_(offset); signs.neg_(); graph.replay()
        _, codes, scales = common.reference_parts(x, signs)
        compare(packed, scale, ref.pack_int4(codes), scales)
    return {'K': k, 'nondefault_stream': True, 'graph_updated_x_and_signs': True}


def run_checks(function, args, report):
    import torch
    from safetensors.torch import load_file
    report['oracle'] = oracle_checks()
    rows = []
    ms = [1, 17, 33] if args.suite == 'smoke' else [1, 7, 17, 31, 32, 33, 4096, 4097]
    for k in [4096, 12288]:
        signs = common.checkpoint_signs(0, 'qkv' if k == 4096 else 'down', 'cuda')
        for m in ms:
            exclusive(args.gpu_info)
            row = {'kind': 'synthetic', **check_case(function, common.make_input(m, k, 'cuda'), signs)}
            rows.append(row); print(json.dumps(row), flush=True)
        report.setdefault('dynamic', []).append(dynamic_checks(function, k))
    fixture_manifest = json.loads((args.fixture_dir / 'manifest.json').read_text())
    assert fixture_manifest['complete'] and fixture_manifest['checkpoint_manifest_sha256'] == report['checkpoint_manifest_sha256']
    fixture_path = args.fixture_dir / 'activations.safetensors'
    assert sha256(fixture_path) == fixture_manifest['data_sha256']
    data = load_file(str(fixture_path), device='cpu')
    for case in fixture_manifest['cases']:
        exclusive(args.gpu_info)
        name = case['name']
        x, signs, packed, scale = [data[f'{name}.{key}'].cuda() for key in ['x', 'signs', 'packed', 'scale']]
        row = {'kind': 'real', 'name': name, **check_case(function, x, signs, (packed, scale))}
        rows.append(row); print(json.dumps(row), flush=True)
    # The harness must fail on a corrupt code, not just report that it ran.
    def corrupt(x, signs, *, out_packed, out_scale):
        common.reference_prepare(x, signs, out_packed=out_packed, out_scale=out_scale)
        out_packed[0, 0].bitwise_xor_(1)
        return out_packed, out_scale
    caught = False
    try:
        check_case(corrupt, common.make_input(1, 4096, 'cuda'), common.checkpoint_signs(0, 'qkv', 'cuda'))
    except AssertionError as error:
        assert 'Quantizer parity failed' in str(error); caught = True
    assert caught
    report.update(checks=rows, fixture_data_sha256=fixture_manifest['data_sha256'],
                  negative_control_rejected=True, complete=True)


def run_timings(function, args, report):
    import torch
    import rht_reference as ref
    from nanovllm.kernels import w4a4_grouped_kernel as gemm
    import kernel_bench_common as timing
    from verify_w4a4_aw_grouped import reference as gemm_reference
    timing.torch = torch
    results = report.setdefault('results', [])
    for m in args.m:
        for name in args.shapes:
            exclusive(args.gpu_info)
            n, k, _, _ = common.SHAPES[name]
            x = common.make_input(m, k, 'cuda')
            signs = common.checkpoint_signs(0, name, 'cuda')
            # Full parity is checked before timing; no checking enters the graph.
            validation = check_case(function, x, signs)
            wp, sw, w_bf16 = common.projection_weights(name, 'cuda')
            ap, sa = common.buffers(x)
            output = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
            bf16_output = torch.empty_like(output)
            def prepare():
                return function(x, signs, out_packed=ap, out_scale=sa)
            def multiply():
                return gemm.w4a4_grouped_gemm(ap, wp, sa, sw, out=output)
            def pipeline():
                prepare(); return multiply()
            pipeline()
            # Sparse numerical audit spans row/channel edges; prepare parity above
            # checks every activation code and scale. Full GEMM tests live in its suite.
            row_ids = torch.linspace(0, m - 1, min(m, 8), device='cuda').round().long().unique()
            col_ids = torch.linspace(0, n - 1, min(n, 64), device='cuda').round().long().unique()
            expected = gemm_reference(ref.unpack_int4(ap[row_ids]), ref.unpack_int4(wp[col_ids]),
                                      sa[row_ids], sw[col_ids]).bfloat16()
            torch.testing.assert_close(output[row_ids][:, col_ids], expected, rtol=0, atol=0)
            count = args.replays or (32 if m == 32 else 4)
            callbacks = {'prepare_only': prepare, 'gemm_only': multiply, 'prepare_plus_gemm': pipeline,
                         'cublas_bf16': lambda: torch.mm(x, w_bf16.T, out=bf16_output)}
            before = timing.telemetry(args.gpu_info)
            graphs = {key: timing.capture(callback, count) for key, callback in callbacks.items()}
            exclusive(args.gpu_info)
            times = timing.measure(graphs, count, args.rounds, 20261108 + m + k + n)
            row = {'name': name, 'M': m, 'N': n, 'K': k, 'validation': validation,
                   'timings': times, 'replays_per_sample': count, 'rounds': args.rounds,
                   'pipeline_audit_rows': row_ids.tolist(), 'pipeline_audit_channels': col_ids.tolist(),
                   'pipeline_sampled_bf16_exact': True,
                   'gemm_config': vars(gemm.best_grouped_config(m, n, k, weight_grouped=True)),
                   'telemetry_before': before, 'telemetry_after': timing.telemetry(args.gpu_info)}
            results.append(row)
            common.atomic_json(args.output_dir / 'timings.json', report)
            print(json.dumps(row), flush=True)
            del graphs, callbacks, x, signs, wp, sw, w_bf16, ap, sa, output, bf16_output
            torch.cuda.empty_cache()
    report['complete'] = True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['check', 'bench'], required=True)
    p.add_argument('--impl', default='reference', help='reference, or module:function implementing the documented adapter')
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--suite', choices=['smoke', 'full'], default='full')
    p.add_argument('--fixture-dir', type=Path, default=common.ROOT / 'fixtures')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--m', type=int, nargs='+', choices=[32, 4096], default=[32, 4096])
    p.add_argument('--shapes', nargs='+', choices=list(common.SHAPES), default=list(common.SHAPES))
    p.add_argument('--rounds', type=int, default=9)
    p.add_argument('--replays', type=int, default=0)
    args = p.parse_args()
    assert args.rounds >= 3 and args.replays >= 0
    args.output_dir = common.output_directory(args.output_dir)
    cache = common.ROOT / 'cache' / f'gpu{args.gpu}'
    os.environ['TILELANG_CACHE_DIR'] = str(cache)
    os.environ['TILELANG_TMP_DIR'] = str(cache / 'tmp')
    os.environ['OMP_NUM_THREADS'] = '1'
    common.checkpoint_manifest()
    args.gpu_info = select_gpu(args.gpu)
    import torch
    torch.set_num_threads(1); torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    function, implementation = common.load_implementation(args.impl)
    report = {'complete': False, 'mode': args.mode, 'implementation': implementation,
              'contract': common.CONTRACT, 'gpu': args.gpu_info, 'torch': torch.__version__,
              'source_sha256': common.source_hashes(),
              'checkpoint_manifest_sha256': sha256(common.CHECKPOINT / 'manifest.json'),
              'timing_scope': 'synthetic BF16 inputs; real layer-0 GPTQ weights; compilation excluded; not engine throughput'}
    path = args.output_dir / ('checks.json' if args.mode == 'check' else 'timings.json')
    try:
        (run_checks if args.mode == 'check' else run_timings)(function, args, report)
        report['peak_cuda_allocated_gib'] = torch.cuda.max_memory_allocated() / 1024**3
    except Exception:
        report['error'] = traceback.format_exc()
        common.atomic_json(path, report)
        raise
    common.atomic_json(path, report)
    print(json.dumps({'complete': True, 'implementation': implementation, 'report': str(path)}), flush=True)


if __name__ == '__main__':
    main()
