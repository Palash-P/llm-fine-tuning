import time
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTTrainer
from trl import SFTConfig


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

TRAIN_FILE = "02_sft/dataset/train.jsonl"
VALIDATION_FILE = "02_sft/dataset/validation.jsonl"

OUTPUT_DIR = "02_sft/output"


# --------------------------------------------------
# Hardware information
# --------------------------------------------------

print("=" * 60)
print("HARDWARE")
print("=" * 60)

print("GPU:", torch.cuda.get_device_name(0))

total_vram = (
    torch.cuda.get_device_properties(0).total_memory
    / 1024**3
)

print(f"Total VRAM: {total_vram:.2f} GB")


# --------------------------------------------------
# Load tokenizer
# --------------------------------------------------

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print("Loading dataset...")

dataset = load_dataset(
    "json",
    data_files={
        "train": TRAIN_FILE,
        "validation": VALIDATION_FILE,
    },
)


print(dataset)


# --------------------------------------------------
# Load model
# --------------------------------------------------

print("\nLoading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
)


print("Model loaded.")


# --------------------------------------------------
# Training configuration
# --------------------------------------------------

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    logging_steps=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    bf16=True,
    report_to="none",
    remove_unused_columns=False,

    # SFT-specific configuration
    max_length=512,
)


# --------------------------------------------------
# SFT Trainer
# --------------------------------------------------

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
)


# --------------------------------------------------
# GPU memory before training
# --------------------------------------------------

torch.cuda.reset_peak_memory_stats()

memory_before = (
    torch.cuda.memory_allocated()
    / 1024**3
)

print("\nGPU memory before training:")
print(f"{memory_before:.2f} GB")


# --------------------------------------------------
# Train
# --------------------------------------------------

print("\nStarting training...")

start_time = time.perf_counter()

trainer.train()

end_time = time.perf_counter()


# --------------------------------------------------
# Training metrics
# --------------------------------------------------

training_time = end_time - start_time

peak_memory = (
    torch.cuda.max_memory_allocated()
    / 1024**3
)


print("\n" + "=" * 60)
print("TRAINING RESULTS")
print("=" * 60)

print(
    f"Training time: "
    f"{training_time:.2f} seconds"
)

print(
    f"Peak GPU memory: "
    f"{peak_memory:.2f} GB"
)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

print("\nRunning evaluation...")

metrics = trainer.evaluate()

print("\nEvaluation metrics:")

for key, value in metrics.items():

    print(f"{key}: {value}")


# --------------------------------------------------
# Save model
# --------------------------------------------------

print("\nSaving model...")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Model saved to: {OUTPUT_DIR}")