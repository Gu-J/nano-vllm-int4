"""Validate fused checkpoint loading, full-model reference and graph replay."""
import argparse
import json
from pathlib import Path
import time

import nano_bench_common as common


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--diagnose', action='store_true')
    args = p.parse_args()
    gpu = common.setup(args.gpu, args.output_dir)
    import torch
    from nanovllm import LLM, SamplingParams
    from nanovllm.layers.linear_w4a4 import W4A4Linear
    from nanovllm.engine.scheduler import Scheduler
    from nanovllm.utils.context import reset_context
    from safetensors import safe_open
    import rht_a4_bench_common as ref_common
    import rht_reference as ref
    from verify_w4a4_aw_grouped import reference as grouped_reference

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    started = time.perf_counter()
    llm = LLM(common.MODEL, w4a4_checkpoint=str(common.CHECKPOINT), max_num_batched_tokens=128,
              max_num_seqs=32, max_model_len=512, gpu_memory_utilization=.45, enforce_eager=False)
    runner = llm.model_runner
    model = runner.model
    modules = [m for m in model.modules() if isinstance(m, W4A4Linear)]
    assert len(modules) == 144 and runner.quantization_info['projections'] == 252
    # Independently audit every concatenation, sign seed and remaining BF16 layer.
    from nanovllm.utils.loader_w4a4 import GROUPS
    manifest = json.loads((common.CHECKPOINT / 'manifest.json').read_text())
    for layer_id, layer in enumerate(model.model.layers):
        with safe_open(str(common.CHECKPOINT / manifest['layers'][str(layer_id)]['file']), framework='pt', device='cpu') as f:
            for group_id, (parent, target, sources, sign) in enumerate(GROUPS):
                module = getattr(getattr(layer, parent), target)
                torch.testing.assert_close(module.signs.cpu(), ref.make_signs(module.in_features, manifest['config']['seed'] + layer_id * 17 + group_id).to(torch.int8), rtol=0, atol=0)
                offset = 0
                for name in sources:
                    packed, scale = f.get_tensor(f'{parent}.{name}.packed'), f.get_tensor(f'{parent}.{name}.scale')
                    n = packed.shape[0]
                    torch.testing.assert_close(module.packed_weight[offset:offset+n].cpu(), packed, rtol=0, atol=0)
                    torch.testing.assert_close(module.weight_scale[offset:offset+n].cpu(), scale, rtol=0, atol=0)
                    offset += n
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    print(json.dumps({'loaded_projection_checks': 252, 'graphs': runner.graph_bs}), flush=True)

    reference_calls = [0]
    native_forward = W4A4Linear.forward
    def reference_forward(self, x):
        flat = x.reshape(-1, self.in_features)
        _, qa, sa = ref_common.reference_parts(flat, self.signs)
        qw = ref.unpack_int4(self.packed_weight)
        reference_calls[0] += 1
        if not args.diagnose:
            # Emulate the specified group-by-group FP32 FMA rounding in FP64.
            # A single dense FP32 GEMM uses a different summation order; even
            # tiny BF16 rounding changes can cross later A4 decision boundaries.
            return grouped_reference(qa, qw, sa, self.weight_scale).bfloat16().reshape(*x.shape[:-1], self.out_features)
        a = (qa.float().view(flat.shape[0], -1, 128) * sa.float().unsqueeze(-1)).reshape_as(flat)
        w = (qw.float().view(self.out_features, -1, 128) * self.weight_scale.float().unsqueeze(-1)).reshape(self.out_features, self.in_features)
        return torch.mm(a, w.T).bfloat16().reshape(*x.shape[:-1], self.out_features)

    def compare(a, b, label, limit=.01):
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        af, bf = a.float(), b.float()
        rrmse = ((af - bf).square().mean() / bf.square().mean()).sqrt().item()
        agreement = (af.argmax(-1) == bf.argmax(-1)).float().mean().item()
        result = {'name': label, 'relative_rmse': rrmse, 'max_abs': (af-bf).abs().max().item(), 'argmax_agreement': agreement}
        print(json.dumps(result), flush=True)
        assert rrmse < limit and agreement >= .9, result
        return result

    inputs = common.prompts(32, 4)
    teacher_tokens = None
    linear_checks = []
    names = {id(m): n for n, m in model.named_modules() if isinstance(m, W4A4Linear)}
    def audit_forward(self, x):
        actual = native_forward(self, x)
        expected = reference_forward(self, x)
        delta = actual.float() - expected.float()
        entry = {'name': names[id(self)], 'M': x.reshape(-1, self.in_features).shape[0],
                 'relative_rmse': (delta.square().mean() / expected.float().square().mean()).sqrt().item(),
                 'max_abs': delta.abs().max().item(), 'unequal_fraction': (actual != expected).float().mean().item()}
        linear_checks.append(entry)
        print(json.dumps({'linear_audit': entry}), flush=True)
        return actual
    def forward_pair(reference=False):
        nonlocal teacher_tokens
        llm.scheduler = Scheduler(llm.config)
        for tokens in inputs:
            llm.add_request(tokens, SamplingParams(max_tokens=8, ignore_eos=True))
        seqs, prefill = llm.scheduler.schedule()
        assert prefill and len(seqs) == 32
        ids, pos = runner.prepare_prefill(seqs)
        runner.enforce_eager = True
        logits = runner.run_model(ids, pos, True).clone()
        if args.diagnose:
            common.save(args.output_dir / 'linear_checks.json', linear_checks)
            return logits, None, {}
        if teacher_tokens is None:
            teacher_tokens = logits.argmax(-1).tolist()
        reset_context()
        llm.scheduler.postprocess(seqs, teacher_tokens, True)
        seqs, prefill = llm.scheduler.schedule()
        assert not prefill
        ids, pos = runner.prepare_decode(seqs)
        eager = runner.run_model(ids, pos, False).clone()
        graphs = {}
        if not reference:
            runner.enforce_eager = False
            for bs in (32, 7, 1):
                ids, pos = runner.prepare_decode(seqs[:bs])
                graph = runner.run_model(ids, pos, False).clone()
                runner.enforce_eager = True
                local_eager = runner.run_model(ids, pos, False).clone()
                graphs[str(bs)] = compare(graph, local_eager, f'graph_vs_eager_b{bs}', limit=.003)
                runner.enforce_eager = False
        reset_context()
        return logits, eager, graphs

    try:
        common.exclusive(gpu)
        with torch.inference_mode():
            if args.diagnose:
                W4A4Linear.forward = audit_forward
            native_pre, native_dec, graph_checks = forward_pair()
            if args.diagnose:
                return
            W4A4Linear.forward = reference_forward
            reference_pre, reference_dec, _ = forward_pair(reference=True)
            W4A4Linear.forward = native_forward
            checks = [compare(native_pre, reference_pre, 'native_vs_grouped_fma_reference_prefill'),
                      compare(native_dec, reference_dec, 'native_vs_grouped_fma_reference_decode')]
            torch.testing.assert_close(native_pre, reference_pre, rtol=0, atol=0)
            torch.testing.assert_close(native_dec, reference_dec, rtol=0, atol=0)
            # Exercise the normal public generation API after restoring native modules.
            runner.enforce_eager = False
            llm.scheduler = Scheduler(llm.config)
            chat = llm.tokenizer.apply_chat_template([{'role': 'user', 'content': 'What is 7 times 8? Answer briefly.'}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            torch.manual_seed(2026)
            generated = llm.generate([chat], SamplingParams(temperature=.1, max_tokens=32), use_tqdm=False)
        result = {'complete': True, 'gpu': gpu, 'loaded_projections': 252, 'native_modules': 144,
                  'reference_linear_calls': reference_calls[0], 'checks': checks, 'graph_checks': graph_checks,
                  'reference': 'Independent PyTorch RHT/quantization, FP32 exact integer group dots, FP64 emulation of every FP32 FMA, final BF16 rounding; full-logit bit equality required.',
                  'generation_smoke': generated, 'elapsed_s': time.perf_counter() - started,
                  'quantization': runner.quantization_info, 'source_sha256': common.source_hashes()}
        common.save(args.output_dir / 'results.json', result)
        print(json.dumps({'complete': True, 'generation_smoke': generated}), flush=True)
    finally:
        W4A4Linear.forward = native_forward
        runner.enforce_eager = False
        llm.exit()


if __name__ == '__main__':
    main()
