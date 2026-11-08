"""Prepare a small, reproducible training-only GPTQ calibration corpus."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import urllib.request

from gsm8k_eval import MODEL
from validate_rht import sha256

URL = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(MODEL))
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent.parent
    assert args.output_dir.resolve().is_relative_to(repo)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_HOME=str(args.output_dir / "cache"), HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    corpus = args.output_dir / "wikitext2_train.txt"
    if not corpus.exists():
        request = urllib.request.Request(URL, headers={"User-Agent": "nano-vllm-gptq-calibration"})
        with urllib.request.urlopen(request, timeout=60) as response:
            contents = response.read(32 * 1024**2 + 1)
        assert 1_000_000 < len(contents) <= 32 * 1024**2
        contents.decode("utf-8")
        corpus.write_bytes(contents)
    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    tokenizer.model_max_length = 10**9
    ids = tokenizer.encode(corpus.read_text(), add_special_tokens=False)
    rng = random.Random(args.seed)
    starts = rng.sample(range(len(ids) - args.length + 1), args.samples)
    tokens = torch.tensor([ids[start:start + args.length] for start in starts], dtype=torch.int64)
    path = args.output_dir / "calibration_tokens.safetensors"
    assert not path.exists(), "Calibration data already exists; use its manifest instead of overwriting"
    save_file({"input_ids": tokens}, str(path))
    manifest = {"corpus": "WikiText-2 train", "url": URL, "corpus_sha256": sha256(corpus),
                "model": str(args.model), "tokenizer_sha256": sha256(args.model / "tokenizer.json"),
                "samples": args.samples, "sequence_length": args.length, "seed": args.seed,
                "corpus_token_count": len(ids), "window_starts": starts,
                "window_overlap_allowed": True, "gsm8k_used": False,
                "tokens_sha256": sha256(path), "source_sha256": sha256(Path(__file__))}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "window_starts"}), flush=True)


if __name__ == "__main__":
    main()
