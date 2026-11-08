"""Resolve benchmark helpers and the canonical nano kernel sources."""
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

KERNEL_FILES = frozenset({
    'rht_a4_tilelang.py', 'rht_a4_configs.json',
    'w4a4_grouped_kernel.py', 'w4a4_aw_grouped_configs.json',
    'w4a4_grouped_configs.json',
})


def source_path(name):
    directory = REPO / 'nanovllm/kernels' if name in KERNEL_FILES else REPO / 'benchmarks'
    return directory / name
