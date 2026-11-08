"""Shared CUDA-graph timing and frozen comparison baselines."""
import importlib.util
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys

from benchmark_utils import source_path
from validate_rht import sha256

SHAPES = {'qkv': (6144, 4096), 'o': (4096, 4096),
          'gate_up': (24576, 4096), 'down': (4096, 12288)}
PTPC_SHA = '104f7a5df30a064f66b6e402e5537c68d98e4d60904521bbee8806a195c705dc'
CONFIG_SHA = 'a196ba88c45432795fbd982dfdb756d472e36af26c8e81eac15bea3c208897b1'


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_ptpc():
    path = Path(__file__).resolve().parent / "baselines/w4a4_ptpc/w4a4_ptpc.py"
    assert sha256(path) == PTPC_SHA
    assert sha256(path.with_name("w4a4_configs.json")) == CONFIG_SHA
    spec = importlib.util.spec_from_file_location("frozen_w4a4_ptpc", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def grouped_quantize(x, group_size, clip):
    m, k = x.shape
    grouped = x.float().reshape(m, k // group_size, group_size)
    maximum = grouped.abs().amax(-1)
    scale = torch.where(maximum == 0, 1., (maximum * clip / 7).clamp_min(torch.finfo(torch.bfloat16).tiny)).bfloat16()
    codes = (grouped / scale.float()[..., None]).round().clamp(-7, 7).to(torch.int8).reshape(m, k)
    return codes.contiguous(), scale.contiguous()


def capture(fn, count):
    for _ in range(8):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(count):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def measure(graphs, count, rounds, seed):
    rng = random.Random(seed)
    times = {key: [] for key in graphs}
    for _ in range(rounds):
        order = list(graphs)
        rng.shuffle(order)
        for name in order:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            times[name].append(start.elapsed_time(end) * 1000 / count)
    return {key: {"median_us": statistics.median(values), "min_us": min(values), "max_us": max(values),
                  "samples_us": values} for key, values in times.items()}


def telemetry(gpu):
    return subprocess.check_output(["nvidia-smi", "-i", gpu["uuid"],
        "--query-gpu=pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw", "--format=csv,noheader,nounits"], text=True).strip()
