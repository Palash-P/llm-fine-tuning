import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print("=" * 60)
print("HARDWARE")
print("=" * 60)

print("GPU:", torch.cuda.get_device_name(0))

total_vram = (
    torch.cuda.get_device_properties(0).total_memory
    / 1024**3
)

print(f"Total VRAM: {total_vram:.2f} GB")

print("\n" + "=" * 60)
print("LOADING MODEL")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
)

print("Model loaded.")

print("\n" + "=" * 60)
print("MODEL PARAMETERS BEFORE LoRA")
print("=" * 60)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

print("\n" + "=" * 60)
print("CREATING LoRA CONFIG")
print("=" * 60)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],

    bias="none",
    task_type="CAUSAL_LM",
)


model = get_peft_model(
    model,
    lora_config,
)


print("\n" + "=" * 60)
print("MODEL PARAMETERS AFTER LoRA")
print("=" * 60)

model.print_trainable_parameters()


total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

trainable_percentage = (
    trainable_params / total_params * 100
)

print(f"\nTotal parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Trainable percentage: {trainable_percentage:.4f}%")