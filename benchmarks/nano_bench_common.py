"""Shared setup for native nano integration checks and latency measurements."""
import hashlib
import json
import os
from pathlib import Path
import sys

from validate_rht import select_gpu, exclusive

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / 'benchmarks/results/nano_integration'
CHECKPOINT = REPO / 'benchmarks/results/rht_gptq_grouped/checkpoint'
MODEL = '/home/huggingface/Qwen3-8B-b968826d9c46dd6066d109eabc6255188de91218'
TOKENS = REPO / 'benchmarks/results/rht_gptq/data/calibration_tokens.safetensors'


def save(path, data):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup(gpu_index, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name in ('TORCH_HOME', 'XDG_CACHE_HOME', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'TILELANG_CACHE_DIR'):
        os.environ[name] = str(ROOT / 'cache' / name.lower())
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
    gpu = select_gpu(gpu_index)
    sys.path.insert(0, str(REPO))
    import torch
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.manual_seed(20261108)
    return gpu


def prompts(batch, length, offset=0):
    # Natural-text token windows already on disk; no dataset download or model copy.
    import torch
    from safetensors.torch import load_file
    manifest = json.loads((CHECKPOINT / 'manifest.json').read_text())
    assert sha(TOKENS) == manifest['config']['calibration']['tokens_sha256']
    data = next(iter(load_file(str(TOKENS), device='cpu').values()))
    rows = []
    for i in range(batch):
        start = (i * 2 + offset) % data.shape[0]
        parts = [data[(start + j) % data.shape[0]] for j in range((length + data.shape[1] - 1) // data.shape[1])]
        rows.append(torch.cat(parts)[:length].tolist())
    assert len({tuple(row[:min(256, length)]) for row in rows}) == batch
    return rows


def source_hashes():
    paths = sorted((REPO / 'nanovllm').rglob('*.py')) + sorted((REPO / 'nanovllm/kernels').glob('*.json'))
    paths += [Path(__file__), Path(__file__).with_name('bench_nano_latency.py'), Path(__file__).with_name('verify_nano_w4a4.py')]
    return {str(p.relative_to(REPO)): sha(p) for p in paths if p.is_file()}
