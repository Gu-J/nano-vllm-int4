from datasets import load_dataset
from gptqmodel import GPTQConfig, GPTQModel

model_id = "Qwen/Qwen3-0.6B"
quant_path = "./downloads/qwen3-0.6b-gptq-int4"

calibration_dataset = load_dataset(
    "wikitext",
    "wikitext-2-raw-v1",
    split="train"
).select(range(128))["text"]

qcfg = GPTQConfig(
    bits=4,
    group_size=128,
    sym=True,     
)

model = GPTQModel.load(model_id, qcfg)

model.quantize(
    calibration_dataset,
    batch_size=1,
)

model.save(quant_path)