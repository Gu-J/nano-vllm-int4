"""Exercise selected native configurations across BF16 exponent extremes."""
from benchmark_utils import source_path
import argparse
import os
from pathlib import Path

import rht_a4_bench_common as common
from validate_rht import select_gpu, exclusive, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    output = common.output_directory(common.REPO / 'benchmarks/results/rht_a4_native/extremes')
    cache = common.REPO / f'benchmarks/results/rht_a4_native/cache/gpu{args.gpu}'
    os.environ['TILELANG_CACHE_DIR'] = str(cache)
    os.environ['TILELANG_TMP_DIR'] = str(cache / 'tmp')
    gpu = select_gpu(args.gpu)
    import torch
    import bench_rht_a4 as checker
    from nanovllm.kernels import rht_a4_tilelang as kernel
    torch.set_num_threads(1); torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    exponents = [-130, -126, -120, -80, -30, 0, 30, 80, 100]
    rows = []
    for k in [4096, 12288]:
        for m in [32, 33]:
            exclusive(gpu)
            generator = torch.Generator(device='cuda').manual_seed(20261108 + m + k)
            x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16, generator=generator).float()
            e = torch.tensor([exponents[i % len(exponents)] for i in range(m)], device='cuda', dtype=torch.int32)
            x = torch.ldexp(x, e[:, None]).bfloat16().contiguous()
            x[-1].zero_()
            assert torch.isfinite(x).all()
            signs = common.checkpoint_signs(0, 'qkv' if k == 4096 else 'down', 'cuda')
            rows.append(checker.check_case(kernel.rht_quant_pack, x, signs))
    common.atomic_json(output / 'results.json', {'complete': True, 'gpu': gpu, 'exponents': exponents,
        'checks': rows, 'kernel_sha256': sha256(Path(kernel.__file__)),
        'configs_sha256': sha256(Path(kernel.__file__).with_name('rht_a4_configs.json'))})
    print({'complete': True, 'checks': len(rows)}, flush=True)


if __name__ == '__main__':
    main()
