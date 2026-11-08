"""Create the RHT/A4 test plan and optionally capture small real-model fixtures."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

import rht_a4_bench_common as common
from validate_rht import select_gpu, exclusive, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=common.ROOT)
    parser.add_argument('--capture-real', action='store_true')
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()
    output = common.output_directory(args.output_dir)
    manifest = common.checkpoint_manifest()
    plan = {
        'contract': common.CONTRACT,
        'checkpoint': str(common.CHECKPOINT),
        'checkpoint_manifest_sha256': sha256(common.CHECKPOINT / 'manifest.json'),
        'source_sha256': common.source_hashes(),
        'performance_M': [32, 4096], 'K': [4096, 12288],
        'correctness_M': [1, 7, 17, 31, 32, 33, 4096, 4097],
        'projections': {name: {'N': item[0], 'K': item[1]} for name, item in common.SHAPES.items()},
        'timing_scopes': ['prepare_only', 'gemm_only', 'prepare_plus_gemm', 'cublas_bf16'],
        'timing_rules': 'warmup/compilation excluded; preallocated outputs; CUDA graphs; raw event samples saved',
        'bf16_baseline': 'cuBLAS torch.mm only; no custom BF16 kernel tuning',
        'split_k': False, 'target': 'one GPU; 32 decode tokens or 4096 prefill tokens',
        'native_operator_status': 'not implemented by this preparation task',
        'fixture_budget_bytes': 64 * 1024**2,
    }
    common.atomic_json(output / 'plan.json', plan)
    print(json.dumps({'plan': str(output / 'plan.json'), 'native_operator_status': plan['native_operator_status']}), flush=True)
    if not args.capture_real:
        return
    directory = output / 'fixtures'; directory.mkdir(exist_ok=True)
    destination = directory / 'activations.safetensors'
    if (directory / 'manifest.json').exists():
        saved = json.loads((directory / 'manifest.json').read_text())
        assert saved['complete'] and saved['checkpoint_manifest_sha256'] == plan['checkpoint_manifest_sha256']
        assert sha256(destination) == saved['data_sha256']
        print(json.dumps({'fixtures_already_complete': len(saved['cases'])}), flush=True)
        return
    assert not destination.exists(), 'Inspect unindexed fixtures before retrying'
    assert shutil.disk_usage(output).free > 2 * 1024**3
    for name in ['HF_HOME', 'HF_HUB_CACHE', 'TORCH_HOME', 'XDG_CACHE_HOME', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR']:
        os.environ[name] = str(output / 'cache' / name.lower())
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
    gpu = select_gpu(args.gpu)
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM
    import rht_gptq_grouped as quant
    import rht_reference as ref
    import a4_group_quant as a4
    torch.set_num_threads(4); torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    started = time.monotonic()
    token_file = common.REPO / 'benchmarks/results/rht_gptq/data/calibration_tokens.safetensors'
    assert sha256(token_file) == manifest['config']['calibration']['tokens_sha256']
    token_state = load_file(str(token_file), device='cpu')
    assert len(token_state) == 1
    tokens = next(iter(token_state.values()))[:32, :32].contiguous()
    assert tokens.shape == (32, 32)
    print(json.dumps({'loading_model': manifest['config']['model'], 'gpu': gpu}), flush=True)
    model = AutoModelForCausalLM.from_pretrained(manifest['config']['model'], local_files_only=True,
                                                torch_dtype=torch.bfloat16, attn_implementation='sdpa').eval().cuda()
    quant.load_checkpoint(model, common.CHECKPOINT, bits=16)
    reload_check = quant.check_reload(model, common.CHECKPOINT)
    audit = quant.set_activation_bits(model, 4)
    captured, metadata, hooks = {}, [], []
    stage = ['prefill']

    def capture(layer_id, group):
        def hook(module, inputs):
            flat = inputs[0].reshape(-1, inputs[0].shape[-1])
            assert flat.dtype == torch.bfloat16 and torch.isfinite(flat).all()
            step = max(1, flat.shape[0] // (128 if stage[0] == 'prefill' else 32))
            indices = torch.arange(0, flat.shape[0], step, device=flat.device)[:128 if stage[0] == 'prefill' else 32]
            name = f'{stage[0]}.layer_{layer_id:02d}.{group}'
            assert name not in captured
            captured[name] = (flat.index_select(0, indices).cpu().contiguous(), module.shared.signs.to('cpu', torch.int8).contiguous())
            metadata.append({'name': name, 'stage': stage[0], 'layer': layer_id, 'group': group,
                             'original_rows': flat.shape[0], 'selected_rows': indices.cpu().tolist(),
                             'M': len(indices), 'K': flat.shape[1]})
        return hook

    for layer_id in [0, 17, 35]:
        for parent, names, group in quant.GROUPS:
            module = getattr(getattr(model.model.layers[layer_id], parent), names[0])
            hooks.append(module.register_forward_pre_hook(capture(layer_id, group)))
    with torch.inference_mode():
        exclusive(gpu)
        prefill = model(tokens.cuda(), use_cache=True, logits_to_keep=1)
        stage[0] = 'decode'
        next_tokens = prefill.logits[:, -1].argmax(-1, keepdim=True)
        decode = model(next_tokens, past_key_values=prefill.past_key_values, use_cache=True, logits_to_keep=1)
        assert torch.isfinite(decode.logits).all() and audit.item() == 0
    for hook in hooks:
        hook.remove()
    assert len(captured) == 24
    del model, prefill, decode, next_tokens
    torch.cuda.empty_cache()
    saved = {'input_ids': tokens}
    for item in metadata:
        exclusive(gpu)
        x, signs = captured[item['name']]
        xx, ss = x.cuda(), signs.cuda()
        rotated, codes, scale = common.reference_parts(xx, ss)
        # Goldens must match the exact backend used for the 89.54% evaluation.
        torch.testing.assert_close(rotated, ref.rht(xx, ss), rtol=0, atol=0)
        eq, es = a4.quantize_rotated(rotated, 128, .9)
        torch.testing.assert_close(codes, eq.reshape_as(codes).to(torch.int8), rtol=0, atol=0)
        torch.testing.assert_close(scale, es, rtol=0, atol=0)
        for key, value in [('x', x), ('signs', signs), ('packed', ref.pack_int4(codes).cpu()), ('scale', scale.cpu())]:
            saved[f"{item['name']}.{key}"] = value.contiguous()
    payload_bytes = sum(t.numel() * t.element_size() for t in saved.values())
    assert payload_bytes < plan['fixture_budget_bytes']
    temporary = destination.with_suffix('.tmp')
    save_file(saved, str(temporary), metadata={'contract': common.CONTRACT['version']})
    temporary.replace(destination)
    record = {'complete': True, 'cases': metadata, 'data_sha256': sha256(destination),
              'bytes': destination.stat().st_size, 'gpu': gpu,
              'checkpoint_manifest_sha256': plan['checkpoint_manifest_sha256'],
              'source_sha256': common.source_hashes(), 'calibration_tokens_sha256': sha256(token_file),
              'prompt': '32 WikiText calibration windows, first 32 tokens each; one greedy decode step',
              'prefill_sampling': '128 evenly spaced rows per selected projection input',
              'layers': [0, 17, 35], 'activation_storage_failures': int(audit.item()),
              'packed_reload_validation': reload_check, 'evaluation_reference_exact': True,
              'elapsed_s': time.monotonic() - started,
              'peak_cuda_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3}
    common.atomic_json(directory / 'manifest.json', record)
    print(json.dumps({'fixtures_complete': len(metadata), 'bytes': record['bytes'], 'elapsed_s': record['elapsed_s']}), flush=True)


if __name__ == '__main__':
    main()
