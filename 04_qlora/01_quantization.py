import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


print("=" * 60)
print("QLoRA QUANTIZATION INSPECTION")
print("=" * 60)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")


print("\nCreating 4-bit quantization configuration...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)


print("\nQuantization configuration:")
print(f"4-bit: {bnb_config.load_in_4bit}")
print(f"Quantization type: {bnb_config.bnb_4bit_quant_type}")
print(f"Double quantization: {bnb_config.bnb_4bit_use_double_quant}")
print(f"Compute dtype: {bnb_config.bnb_4bit_compute_dtype}")


print("\nLoading quantized model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

print("Model loaded.")


print("\nModel dtype information:")

for name, param in list(model.named_parameters())[:10]:
    print(
        f"{name:60} "
        f"dtype={str(param.dtype):10} "
        f"device={param.device}"
    )


print("\nGPU memory after loading:")
print(
    f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
)