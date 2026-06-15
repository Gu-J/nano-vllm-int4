# Nano-vLLM-INT4

A lightweight INT4 inference implementation based on Nano-vLLM and TileLang.

This project extends Nano-vLLM with support for INT4 weight-only quantization and implements a custom W4A16 GEMM kernel using TileLang.

This project is primarily intended for learning and experimentation.

## Features

* INT4 weight loading and inference support
* W4A16 GEMM kernel implemented with TileLang
* INT4 fast dequantization and Split-K optimization
* Qwen3 INT4 deployment pipeline
* End-to-end FP16 → INT4 quantization and inference

## Implementation

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
