import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    use_w4a16: bool = False
    w4a4_checkpoint: str | None = None

    def __post_init__(self):
        # assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        if self.w4a4_checkpoint is not None:
            if self.use_w4a16:
                raise ValueError("W4A16 and W4A4 cannot be enabled together")
            if self.tensor_parallel_size != 1:
                raise ValueError("RHT W4A4 currently supports one GPU only")
            if not os.path.isfile(os.path.join(self.w4a4_checkpoint, "manifest.json")):
                raise ValueError("w4a4_checkpoint must contain manifest.json")
