"""Load the calibrated RHT/G128 checkpoint without materializing BF16 linears."""
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from huggingface_hub import snapshot_download

from nanovllm.layers.linear_w4a4 import W4A4Linear
from nanovllm.utils.loader import default_weight_loader


FORMAT = "rht_gptq_static_group128_bf16_scale_i4_v1"
GROUPS = (
    ("self_attn", "qkv_proj", ("q_proj", "k_proj", "v_proj"), "qkv"),
    ("self_attn", "o_proj", ("o_proj",), "o"),
    ("mlp", "gate_up_proj", ("gate_proj", "up_proj"), "gate_up"),
    ("mlp", "down_proj", ("down_proj",), "down"),
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def load_w4a4_model(model, model_path, checkpoint_path):
    base = Path(model_path)
    if not base.is_dir():
        base = Path(snapshot_download(repo_id=model_path, local_files_only=True))
    checkpoint = Path(checkpoint_path).resolve()
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    cfg = manifest["config"]
    if manifest.get("format") != FORMAT or not manifest.get("complete"):
        raise ValueError("Incomplete or unsupported W4A4 checkpoint")
    if (cfg.get("group_size") != 128 or not cfg.get("static_groups")
            or cfg.get("runtime_column_permutation") is not False
            or cfg.get("evaluation_activation_group_size") != 128
            or cfg.get("evaluation_activation_clipping") != 0.9
            or cfg.get("layout_version") != "row_DH_paley12_kron_sylvester_v1"):
        raise ValueError("Checkpoint quantization contract differs from the native kernel")
    if sha256(base / "config.json") != cfg["model_config_sha256"]:
        raise ValueError("Base model config does not match the calibrated checkpoint")
    layers = model.model.layers
    if set(manifest["layers"]) != {str(i) for i in range(len(layers))}:
        raise ValueError("Checkpoint layer set does not match the model")

    # Only embeddings, head and norms are parameters in the quantized model.
    # Skip original projection weights before reading their tensor payloads.
    remaining = dict(model.named_parameters())
    for shard in sorted(base.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in remaining:
                    param = remaining.pop(name)
                    value = handle.get_tensor(name)
                    if value.dtype != torch.bfloat16 or param.dtype != torch.bfloat16:
                        raise ValueError(f"Expected BF16 base parameter: {name}")
                    getattr(param, "weight_loader", default_weight_loader)(param, value)
    if remaining:
        raise ValueError(f"Missing base parameters: {sorted(remaining)}")

    count = 0
    for index, layer in enumerate(layers):
        entry = manifest["layers"][str(index)]
        shard = (checkpoint / entry["file"]).resolve()
        if not shard.is_relative_to(checkpoint) or sha256(shard) != entry["sha256"]:
            raise ValueError(f"Invalid checkpoint shard: layer {index}")
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for parent, target, sources, sign_name in GROUPS:
                module = getattr(getattr(layer, parent), target)
                if not isinstance(module, W4A4Linear):
                    raise TypeError(f"Expected W4A4Linear at layer {index}/{target}")
                signs = handle.get_tensor(f"signs.{sign_name}")
                if (signs.dtype != torch.int8 or signs.shape != module.signs.shape
                        or not torch.all((signs == 1) | (signs == -1))):
                    raise ValueError(f"Invalid RHT signs at layer {index}/{target}")
                packed = torch.cat([handle.get_tensor(f"{parent}.{name}.packed") for name in sources])
                scale = torch.cat([handle.get_tensor(f"{parent}.{name}.scale") for name in sources])
                if (packed.dtype != torch.int8 or packed.shape != module.packed_weight.shape
                        or scale.dtype != torch.bfloat16 or scale.shape != module.weight_scale.shape
                        or not torch.isfinite(scale).all() or not torch.all(scale > 0)):
                    raise ValueError(f"Invalid weight/scale at layer {index}/{target}")
                module.packed_weight.copy_(packed)
                module.weight_scale.copy_(scale)
                module.signs.copy_(signs)
                count += len(sources)
    return {"format": FORMAT, "projections": count, "native_linear_modules": len(layers) * 4,
            "checkpoint_manifest_sha256": sha256(checkpoint / "manifest.json"),
            "weight_group_size": 128, "activation_group_size": 128, "activation_clipping": 0.9,
            "scale_dtype": "bfloat16", "split_k": False}
