# Nano-vLLM-INT4

A lightweight INT4 inference implementation based on Nano-vLLM and TileLang.

The current path runs Qwen3-8B with RHT-aware GPTQ W4A4 on one GPU. Both
weights and activations use groups of 128 and BF16 scales; activation clipping
is 0.9. TileLang fuses RHT, activation quantization and packing, then executes
a native INT4 GEMM with fused QKV and gate/up projections and no Split-K.

This project is primarily intended for learning and experimentation.

## Qwen3-8B W4A4

Use the existing BF16 model directory for the tokenizer, config, embeddings,
norms and LM head, and the calibrated checkpoint for packed projection weights:

```python
from nanovllm import LLM, SamplingParams

model = "/home/huggingface/Qwen3-8B-b968826d9c46dd6066d109eabc6255188de91218"
llm = LLM(
    model,
    w4a4_checkpoint="benchmarks/results/rht_gptq_grouped/checkpoint",
    tensor_parallel_size=1,
    max_num_seqs=32,
    max_num_batched_tokens=4096,
    max_model_len=4352,
    enforce_eager=False,
)
outputs = llm.generate(["Explain the purpose of a GPU."], SamplingParams(max_tokens=128))
llm.exit()
```

Omit `w4a4_checkpoint` to use the default BF16/cuBLAS path. CUDA Graphs support
batch sizes up to `max_num_seqs`, including batch 32. Attention and KV cache
remain BF16. No full BF16 copy of quantized projection weights is loaded.

Run the latency comparison on an idle GPU, using a fresh output directory:

```bash
python benchmarks/bench_nano_latency.py --backend bf16 --gpu 1 \
  --output-dir benchmarks/results/nano_integration/bf16_new
python benchmarks/bench_nano_latency.py --backend w4a4 --gpu 1 \
  --output-dir benchmarks/results/nano_integration/w4a4_new
```

The default benchmark uses 32 requests, 4096 input tokens and 256 output tokens,
serial prefill followed by batch-32 decode. TTFT includes queueing; individual
prefill service time and decode-only ITL are also recorded. Each backend uses
three measured rounds after warmup with prefix reuse disabled.

See [calibration, accuracy and performance benchmarks](benchmarks/README.md).
The native W4A4 path scores 1200/1319 (90.9780%) on the full GSM8K test set;
the benchmark README records the evaluation protocol and historical comparison.

## Earlier W4A16 implementation

The original W4A16 path remains available through `use_w4a16=True`.

### Features

* INT4 weight loading and inference support
* W4A16 GEMM kernel implemented with TileLang
* INT4 fast dequantization and Split-K optimization
* Qwen3 INT4 deployment pipeline
* End-to-end FP16 → INT4 quantization and inference

### Implementation

Compared with the original Nano-vLLM, the following components are added:

* `nanovllm/layers/linear_w4a16.py`

  * TileLang W4A16 GEMM kernel
  * INT4 fast dequantization and Split-K optimization 
  * Modified linear layer for quantized weights

* `nanovllm/models/qwen3_w4a16.py`

  * Qwen3 model implementation with INT4 support

* `gptqmodel_quant.py`

  * Example script for quantizing FP16 models into INT4 format

* `download_model.py`

  * Download and convert pre-quantized models from Hugging Face

## Acknowledgements

* Nano-vLLM
* TileLang
* Qwen3
