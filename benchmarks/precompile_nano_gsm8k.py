"""Populate unchanged native GEMM caches for known GSM8K prefill shapes."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def compile_one(shape):
    import torch
    torch.set_num_threads(1)
    from nanovllm.kernels.w4a4_grouped_kernel import best_grouped_config, compile_grouped_gemm
    m, n, k = shape
    config = best_grouped_config(m, n, k, weight_grouped=True)
    start = time.monotonic()
    kernel = compile_grouped_gemm(m, n, k, config, weight_grouped=True)
    return {'shape': shape, 'config': asdict(config), 'pid': os.getpid(), 'seconds': time.monotonic()-start,
            'cuda_sha256': hashlib.sha256(kernel.get_kernel_source().encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    from validate_rht import select_gpu
    from gsm8k_gptq_grouped_eval import atomic_json
    gpu = select_gpu(args.gpu)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name in ('TORCH_HOME','XDG_CACHE_HOME','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','TILELANG_CACHE_DIR'):
        os.environ[name] = str(REPO / 'benchmarks/results/nano_integration/cache' / name.lower())
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                      OMP_NUM_THREADS='1', MAX_JOBS='1')
    import torch
    torch.set_num_threads(1)
    from transformers import AutoTokenizer
    from gsm8k_eval import prepare_data, MODEL
    from nanovllm.engine.scheduler import Scheduler
    from nanovllm.engine.sequence import Sequence
    from nanovllm import SamplingParams
    rows, prefix, _ = prepare_data(REPO / 'benchmarks/results/gsm8k_rht/data')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    prompts = []
    for index, row in enumerate(rows):
        prompt = tokenizer.apply_chat_template([{'role':'user','content':prefix + f"\n\nQ: {row['question']}\nA:"}],
                                               tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompts.append((len(ids), index, ids))
    prompts.sort(key=lambda r:(r[0],r[1]))
    cfg = SimpleNamespace(max_num_seqs=32,max_num_batched_tokens=4096,eos=151645,
                          kvcache_block_size=256,num_kvcache_blocks=1029)
    plan = []
    for start in range(0,len(prompts),32):
        scheduler = Scheduler(cfg)
        for _, _, ids in prompts[start:start+32]:
            scheduler.add(Sequence(ids, SamplingParams(max_tokens=1024)))
        shapes = []
        while scheduler.waiting:
            seqs, prefill = scheduler.schedule()
            assert prefill
            shapes.append(sum(seq.num_scheduled_tokens for seq in seqs))
            # Prefill block hashes depend only on prompt tokens, not this dummy
            # first output token. No decode or model execution is simulated.
            scheduler.postprocess(seqs, [0] * len(seqs), True)
        plan.append({'batch': start//32, 'ids':[r[1] for r in prompts[start:start+32]], 'M':shapes})
    evaluation = REPO / 'benchmarks/results/gsm8k_nano/full_v1'
    eval_config = json.loads((evaluation/'config.json').read_text())
    assert [r['ids'] for r in plan] == eval_config['ordered_batch_ids']
    for batch in plan:
        path = evaluation/'batch_audits'/f"batch_{batch['batch']:03d}.json"
        if path.is_file(): assert len(batch['M'])==json.loads(path.read_text())['prefill_steps']
    completed = json.loads((evaluation/'progress.json').read_text()).get('global_batch_index',-1)
    dimensions = [(6144,4096),(4096,4096),(24576,4096),(4096,12288)]
    tasks, seen = [], set()
    for batch in plan:
        if batch['batch'] <= completed + 1: continue
        for m in batch['M']:
            for n,k in dimensions:
                shape = (m,n,k)
                if shape not in seen:
                    seen.add(shape); tasks.append(shape)
    report = {'complete':False,'gpu':gpu,'workers':args.workers,'plan':plan,'kernel_tasks':len(tasks),
              'note':'Compile only; original native module/config selection. TileLang publishes complete disk cache directories atomically.',
              'completed':[]}
    atomic_json(args.output_dir/'results.json',report)
    print(json.dumps({'planned_batches':len(plan),'kernel_tasks':len(tasks),'gpu':gpu}),flush=True)
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as executor:
        futures = [executor.submit(compile_one, shape) for shape in tasks]
        for future in as_completed(futures):
            report['completed'].append(future.result())
            if len(report['completed']) % 8 == 0:
                report['elapsed_s']=time.monotonic()-started
                atomic_json(args.output_dir/'results.json',report)
                print(json.dumps({'compiled':len(report['completed']),'total':len(tasks),'elapsed_s':round(report['elapsed_s'],1)}),flush=True)
    report.update(complete=True,elapsed_s=time.monotonic()-started)
    atomic_json(args.output_dir/'results.json',report)
    print(json.dumps({'complete':True,'compiled':len(tasks),'elapsed_s':report['elapsed_s']}),flush=True)


if __name__=='__main__':
    main()
