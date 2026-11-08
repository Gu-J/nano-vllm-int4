# Benchmarks

The maintained path is Qwen3-8B RHT-aware GPTQ W4A4 on one GPU: W-G128,
A-G128, activation clipping 0.9, BF16 scales, native INT4 MMA and no Split-K.
Benchmark scripts import the production kernels and dispatch tables from
`nanovllm/kernels/`. Frozen PTPC and activation-only G128 kernels in `baselines/`
are retained solely for speed comparisons; the BF16 baseline is cuBLAS.

Run commands from the repository root in the `nano` environment. Select an idle
GPU with `nvidia-smi`; GPU-taking entry points reject a busy device. Use a fresh
output directory for each measurement. Model, dataset, JIT-cache and result
artifacts live under the ignored `results/` directory and are not included in Git.

## Checkpoint and calibration

The retained checkpoint is `results/rht_gptq_grouped/checkpoint` (about 3.34 GiB).
It contains 36 shards / 252 quantized projections, fixed RHT signs, BF16 scales
and a provenance manifest. The original BF16 Qwen3-8B directory supplies the
tokenizer, config, embeddings, norms and LM head. This is not a standalone
Hugging Face checkpoint.

The calibration uses 128 WikiText-2 train sequences of 2048 tokens, seed 2026,
static groups in original rotated K order, act-order and 1% Hessian damping.
BF16 scales are rounded before GPTQ decisions. A4 is disabled during Hessian
collection; calibration propagates quantized weights layer by layer.

```bash
python benchmarks/prepare_gptq_data.py \
  --output-dir benchmarks/results/rht_gptq/data
python benchmarks/calibrate_rht_gptq_grouped.py --gpu 1 \
  --data-dir benchmarks/results/rht_gptq/data \
  --output-dir benchmarks/results/rht_gptq_grouped/checkpoint_new
python benchmarks/verify_rht_gptq_grouped.py --gpu 1
```

The existing final checkpoint is reused by default. `checkpoint_new` is only
for intentional recalibration; checks and model examples point to the retained
`checkpoint`. The small `results/rht_gptq/data` directory is shared calibration
data, not the retired per-channel quantized model.

## Native inference latency

```bash
python benchmarks/verify_nano_w4a4.py --gpu 1 \
  --output-dir benchmarks/results/nano_integration/recheck
python benchmarks/bench_nano_latency.py --backend bf16 --gpu 1 \
  --output-dir benchmarks/results/nano_integration/bf16_new
python benchmarks/bench_nano_latency.py --backend w4a4 --gpu 1 \
  --output-dir benchmarks/results/nano_integration/w4a4_new
```

Recorded on the same RTX A6000 48 GB: 32 simultaneous requests, 4096 input
tokens each, serial prefill, 256 output tokens, and CUDA Graph decode. Three
measured rounds per backend exclude model loading, JIT and warmup; each round
starts with fresh scheduler state. ITL covers 255 pure decode steps per round,
including the host scheduler and token return, with prefill excluded.

| Metric | BF16/cuBLAS | Native W4A4 |
| --- | ---: | ---: |
| Mean prefill service time | 585.31 ms | 361.85 ms |
| Mean TTFT, including queueing | 9656.54 ms | 5976.44 ms |
| TTFT p95 | 18077.98 ms | 11182.93 ms |
| Mean decode ITL | 56.68 ms | 41.74 ms |
| Decode ITL p95 | 57.43 ms | 42.56 ms |
| Aggregate decode tokens/s | 564.57 | 766.71 |

TTFT improves 1.62x and ITL 1.36x for this workload. Model tensors occupy
15.28 GiB in BF16 and 5.67 GiB in W4A4; automatic KV-cache sizing uses the
remaining memory, so total reserved VRAM is not a weight-memory comparison.
Raw measurements remain in `results/nano_integration/`.

## GSM8K accuracy

```bash
python benchmarks/gsm8k_nano_eval.py --gpu 1 \
  --output-dir benchmarks/results/gsm8k_nano/new_run
```

The full native run scores **1200/1319 (90.9780%)** with both strict and flexible
extraction. It uses eight CoT demonstrations, the Qwen chat template, thinking
disabled, greedy argmax and a 1024-token generation cap. Prompts are sorted into
groups of 32; the native scheduler handles packed prefill and shrinking decode
batches. The evaluation adapter stops on either EOS 151643 or 151645. There
were four length-capped generations and five strict extraction failures.

The earlier Transformers/SDPA BF16 run scored 1231/1319 (93.3283%); the native
score is 2.3503 percentage points lower. This is a historical backend comparison,
not a newly measured nano BF16 accuracy result. G128/clipping=0.9 was selected
using a prior GSM8K subset, which is included in the full score.

The evaluator deliberately checks the saved dataset, prompt hashes and historical
protocol before running. It requires local `results/gsm8k_rht/{data,bf16}` and
`results/gsm8k_gptq_grouped/worker_0/config.json`, in addition to the final
checkpoint. These experiment artifacts are ignored and are not supplied by a
fresh clone. `precompile_nano_gsm8k.py` is an optional helper for the saved
evaluation shapes. Original predictions, audits and summaries remain under
`results/gsm8k_nano/`.

## Operators and tuning

```bash
python benchmarks/prepare_rht_a4_bench.py --capture-real --gpu 1
python benchmarks/bench_rht_a4.py --mode check --gpu 1 \
  --impl nanovllm.kernels.rht_a4_tilelang:rht_quant_pack \
  --output-dir benchmarks/results/rht_a4_native/recheck
python benchmarks/bench_rht_a4.py --mode bench --gpu 1 \
  --impl nanovllm.kernels.rht_a4_tilelang:rht_quant_pack \
  --output-dir benchmarks/results/rht_a4_native/remeasure
python benchmarks/tune_rht_a4.py --gpu 1 \
  --output-dir benchmarks/results/rht_a4_native/retune
python benchmarks/verify_w4a4_aw_grouped.py --gpu 1
python benchmarks/tune_w4a4_aw_grouped.py --gpu 1 --m 32 4096 --tune \
  --output-dir benchmarks/results/w4a4_aw_grouped/retune
```

The GEMM tuner now defaults to nano's four fused shapes: QKV (6144,4096), O
(4096,4096), gate/up (24576,4096), and down (4096,12288), expressed as (N,K).
Only W4A4 is tuned. Configurations are written to the selected result directory;
promoting them to production dispatch tables is a separate step.

Recorded tuned operator measurements (microseconds, layer-0 weights, reused
buffers, randomized CUDA Graph timing; these are not a whole-model profile):

| M | Projection | RHT + quant + pack | GEMM | Full chain | cuBLAS BF16 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 32 | QKV | 6.91 | 21.25 | 28.12 | 77.72 |
| 32 | O | 6.92 | 14.97 | 21.87 | 52.41 |
| 32 | Gate/up | 6.91 | 79.86 | 86.85 | 323.27 |
| 32 | Down | 17.99 | 46.06 | 64.65 | 167.13 |
| 4096 | QKV | 155.40 | 861.77 | 1022.30 | 1964.69 |
| 4096 | O | 146.93 | 505.62 | 656.60 | 1289.60 |
| 4096 | Gate/up | 133.89 | 3293.68 | 3482.32 | 6765.90 |
| 4096 | Down | 635.00 | 1550.64 | 2194.38 | 3595.79 |

RHT validation checks exact packed codes and BF16 scales, tails, canaries,
updated graph inputs and CUDA streams. GEMM validation uses an independent
per-group FMA reference. Quantization and reduction order are part of the
numerical contract; a dense dequantized GEMM has a different accumulation order.

The retained Python reference and GPTQ helper modules are calibration and
correctness dependencies. Historical sweep launchers, packaging scripts,
intermediate reports and duplicate production kernels are excluded from this
maintained benchmark tree; raw evidence and a pre-cleanup source archive remain
under ignored `results/`.
