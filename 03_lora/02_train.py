import time
import torch

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

TRAIN_FILE = "02_sft/dataset/train.jsonl"
VALIDATION_FILE = "02_sft/dataset/validation.jsonl"

OUTPUT_DIR = "03_lora/output_v5"


print("=" * 60)
print("HARDWARE")
print("=" * 60)

print("GPU:", torch.cuda.get_device_name(0))

total_vram = (
    torch.cuda.get_device_properties(0).total_memory
    / 1024**3
)

print(f"Total VRAM: {total_vram:.2f} GB")


# ---------------------------------------------------------
# Load tokenizer
# ---------------------------------------------------------

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

print("Loading dataset...")

dataset = load_dataset(
    "json",
    data_files={
        "train": TRAIN_FILE,
        "validation": VALIDATION_FILE,
    },
)

print(dataset)


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

print("\nLoading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
)

print("Model loaded.")


# ---------------------------------------------------------
# LoRA configuration
# ---------------------------------------------------------

print("\nCreating LoRA configuration...")

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],

    bias="none",
    task_type="CAUSAL_LM",
)


# ---------------------------------------------------------
# Training configuration
# ---------------------------------------------------------

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,

    num_train_epochs=3,

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,

    gradient_accumulation_steps=4,

    learning_rate=1e-4,

    logging_steps=1,

    eval_strategy="epoch",
    save_strategy="epoch",

    bf16=True,

    report_to="none",

    remove_unused_columns=False,

    max_length=512,
)


# ---------------------------------------------------------
# Create trainer
# ---------------------------------------------------------

trainer = SFTTrainer(
    model=model,

    args=training_args,

    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],

    processing_class=tokenizer,

    peft_config=lora_config,
)


# ---------------------------------------------------------
# Parameter statistics
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("TRAINABLE PARAMETERS")
print("=" * 60)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

print(
    f"Trainable percentage: "
    f"{trainable_params / total_params * 100:.4f}%"
)


# ---------------------------------------------------------
# GPU memory before training
# ---------------------------------------------------------

torch.cuda.reset_peak_memory_stats()

memory_before = (
    torch.cuda.memory_allocated() / 1024**3
)

print("\nGPU memory before training:")
print(f"{memory_before:.2f} GB")


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

print("\nStarting LoRA training...")

start_time = time.perf_counter()

trainer.train()

end_time = time.perf_counter()

training_time = end_time - start_time

peak_memory = (
    torch.cuda.max_memory_allocated()
    / 1024**3
)


# ---------------------------------------------------------
# Training results
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("LORA TRAINING RESULTS")
print("=" * 60)

print(f"Training time: {training_time:.2f} seconds")
print(f"Peak GPU memory: {peak_memory:.2f} GB")


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

print("\nRunning evaluation...")

metrics = trainer.evaluate()

print("\nEvaluation metrics:")

for key, value in metrics.items():
    print(f"{key}: {value}")


# ---------------------------------------------------------
# Save adapter
# ---------------------------------------------------------

print("\nSaving LoRA adapter...")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"LoRA adapter saved to: {OUTPUT_DIR}")