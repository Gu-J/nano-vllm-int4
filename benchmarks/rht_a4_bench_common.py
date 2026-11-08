"""Frozen RHT/A4 test contract and a reference-only implementation adapter."""
from benchmark_utils import source_path
import importlib
import json
from pathlib import Path

from validate_rht import sha256

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / 'benchmarks/results/rht_a4_preparation'
CHECKPOINT = REPO / 'benchmarks/results/rht_gptq_grouped/checkpoint'
GROUP_SIZE, CLIP = 128, 0.9
SHAPES = {
    'qkv': (6144, 4096, 'self_attn', ['q_proj', 'k_proj', 'v_proj']),
    'o': (4096, 4096, 'self_attn', ['o_proj']),
    'gate_up': (24576, 4096, 'mlp', ['gate_proj', 'up_proj']),
    'down': (4096, 12288, 'mlp', ['down_proj']),
}
CONTRACT = {
    'version': 'rht_a4_g128_c90_packed_v1',
    'layout': 'row_DH_paley12_kron_sylvester_v1',
    'input': 'contiguous BF16 X[M,K], K=4096 or 12288',
    'signs': 'contiguous INT8 D[K], values {-1,+1}, fixed by checkpoint',
    'rotation': 'X D H; FP32 intermediates; normalized H4096 or H12 tensor H1024',
    'group_size': GROUP_SIZE, 'clipping': CLIP,
    'scale': 'BF16(max(abs(rotated_group))*0.9/7), rounded BEFORE codes',
    'zero_group_scale': 1.0,
    'minimum_nonzero_scale': 'torch.finfo(torch.bfloat16).tiny',
    'rounding': 'nearest, ties to even', 'code_range': [-7, 7],
    'packing': 'two signed twos-complement INT4 per byte; even K in low nibble',
    'outputs': 'INT8 packed[M,K/2], BF16 scale[M,K/128], caller-owned contiguous buffers',
    'api': 'fn(x, signs, *, out_packed, out_scale) -> (out_packed, out_scale)',
    'runtime': 'current CUDA stream; no host synchronization; graph replay reads updated inputs/signs',
    'mutation': 'inputs, signs and memory outside output slices must remain unchanged',
    'nonfinite_inputs': 'outside this contract; finite BF16 inputs with finite FP32 RHT required',
}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def output_directory(path):
    path = Path(path).resolve()
    if not path.is_relative_to(REPO / 'benchmarks/results'):
        raise ValueError('Outputs must be inside benchmarks/results')
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_manifest():
    manifest = json.loads((CHECKPOINT / 'manifest.json').read_text())
    assert manifest['complete'] and manifest['projections'] == 252
    cfg = manifest['config']
    assert cfg['group_size'] == cfg['evaluation_activation_group_size'] == GROUP_SIZE
    assert cfg['evaluation_activation_clipping'] == CLIP and cfg['static_groups']
    assert cfg['layout_version'] == CONTRACT['layout']
    for name, digest in cfg['source_sha256'].items():
        assert sha256(source_path(name)) == digest, name
    return manifest


def source_hashes():
    names = ['rht_a4_bench_common.py', 'prepare_rht_a4_bench.py', 'bench_rht_a4.py',
             'rht_reference.py', 'a4_group_quant.py', 'rht_gptq_grouped.py',
             'w4a4_grouped_kernel.py', 'w4a4_aw_grouped_configs.json',
             'verify_w4a4_aw_grouped.py', 'kernel_bench_common.py']
    return {name: sha256(source_path(name)) for name in names}


def reference_parts(x, signs):
    import torch
    import a4_group_quant as a4
    # Existing evaluation path caches H12 on device and supports CUDA graphs.
    rotated = a4.rotate(x, signs)
    codes, scales = a4.quantize_rotated(rotated, GROUP_SIZE, CLIP)
    return rotated, codes.reshape(x.shape).to(dtype=torch.int8), scales


def reference_prepare(x, signs, *, out_packed, out_scale):
    import rht_reference as ref
    _, codes, scales = reference_parts(x, signs)
    out_packed.copy_(ref.pack_int4(codes))
    out_scale.copy_(scales)
    return out_packed, out_scale


def load_implementation(spec):
    if spec == 'reference':
        return reference_prepare, {'name': spec, 'reference_only': True, 'native_kernel_tested': False}
    module_name, function_name = spec.split(':', 1)
    module = importlib.import_module(module_name)  # No implicit reference fallback.
    function = getattr(module, function_name)
    assert callable(function)
    return function, {'name': spec, 'reference_only': False,
                      'module_path': str(Path(module.__file__).resolve()),
                      'module_sha256': sha256(Path(module.__file__))}


def make_input(m, k, device, seed=20261108):
    import torch
    generator = torch.Generator(device=device).manual_seed(seed + m * 17 + k)
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16, generator=generator)
    # Every large shape also covers zeros, outliers, constants and exponent range.
    x[0].zero_()
    if m > 1:
        x[1].zero_(); x[1, k // 12 if k == 12288 else 5] = 1000
    if m > 2:
        x[2].fill_(1)
    if m > 3:
        x[3, ::2] = 1; x[3, 1::2] = -1
    if m > 4:
        x[4].mul_(1e-9)
    if m > 5:
        x[5].mul_(1e4)
    return x


def buffers(x):
    import torch
    m, k = x.shape
    return (torch.empty((m, k // 2), device=x.device, dtype=torch.int8),
            torch.empty((m, k // GROUP_SIZE), device=x.device, dtype=torch.bfloat16))


def checkpoint_signs(layer, group, device):
    import torch
    from safetensors import safe_open
    with safe_open(str(CHECKPOINT / f'weights/layer_{layer:02d}.safetensors'), framework='pt', device='cpu') as handle:
        return handle.get_tensor(f'signs.{group}').to(device=device, dtype=torch.int8).contiguous()


def projection_weights(shape, device):
    """Load layer-0 packed GPTQ weights and the original BF16 cuBLAS operand."""
    import torch
    from safetensors import safe_open
    n, k, parent, names = SHAPES[shape]
    manifest = checkpoint_manifest()
    shard = CHECKPOINT / manifest['layers']['0']['file']
    assert sha256(shard) == manifest['layers']['0']['sha256']
    with safe_open(str(shard), framework='pt', device='cpu') as handle:
        packed = torch.cat([handle.get_tensor(f'{parent}.{name}.packed') for name in names])
        scale = torch.cat([handle.get_tensor(f'{parent}.{name}.scale') for name in names])
    model = Path(manifest['config']['model'])
    index = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
    originals = []
    for name in names:
        key = f'model.layers.0.{parent}.{name}.weight'
        with safe_open(str(model / index[key]), framework='pt', device='cpu') as handle:
            originals.append(handle.get_tensor(key))
    original = torch.cat(originals)
    assert packed.shape == (n, k // 2) and scale.shape == (n, k // 128)
    assert original.shape == (n, k) and original.dtype == torch.bfloat16
    return packed.to(device), scale.to(device), original.to(device)
