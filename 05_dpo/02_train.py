import time
import torch

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

TRAIN_FILE = "05_dpo/dataset/train.jsonl"
VALIDATION_FILE = "05_dpo/dataset/validation.jsonl"

OUTPUT_DIR = "05_dpo/output_v1"


# ============================================================
# HARDWARE
# ============================================================

print("=" * 60)
print("HARDWARE")
print("=" * 60)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(
    f"Total VRAM: "
    f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
)


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# DATASET
# ============================================================

print("Loading DPO dataset...")

dataset = load_dataset(
    "json",
    data_files={
        "train": TRAIN_FILE,
        "validation": VALIDATION_FILE,
    },
)

print(dataset)


# ============================================================
# QUANTIZATION
# ============================================================

print("\nCreating 4-bit quantization configuration...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading 4-bit model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

print("Model loaded.")


# ============================================================
# LoRA CONFIGURATION
# ============================================================

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


# ============================================================
# DPO CONFIGURATION
# ============================================================

print("\nCreating DPO configuration...")

training_args = DPOConfig(
    output_dir=OUTPUT_DIR,

    num_train_epochs=3,

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,

    gradient_accumulation_steps=4,

    learning_rate=1e-5,

    beta=0.1,

    logging_steps=1,

    eval_strategy="epoch",
    save_strategy="epoch",

    bf16=True,

    report_to="none",

    max_length=512,
)


# ============================================================
# DPO TRAINER
# ============================================================

print("\nCreating DPO trainer...")

trainer = DPOTrainer(
    model=model,
    ref_model=None,

    args=training_args,

    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],

    processing_class=tokenizer,

    peft_config=lora_config,
)


# ============================================================
# MEMORY
# ============================================================

torch.cuda.reset_peak_memory_stats()

print("\nGPU memory before training:")

print(
    f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
)


# ============================================================
# TRAIN
# ============================================================

print("\nStarting DPO training...")

start_time = time.time()

trainer.train()

training_time = time.time() - start_time


# ============================================================
# RESULTS
# ============================================================

peak_memory = (
    torch.cuda.max_memory_allocated()
    / 1024**3
)

print("\n" + "=" * 60)
print("DPO TRAINING RESULTS")
print("=" * 60)

print(f"Training time: {training_time:.2f} seconds")
print(f"Peak GPU memory: {peak_memory:.2f} GB")


# ============================================================
# EVALUATION
# ============================================================

print("\nRunning evaluation...")

metrics = trainer.evaluate()

print("\nEvaluation metrics:")

for key, value in metrics.items():
    print(f"{key}: {value}")


# ============================================================
# SAVE
# ============================================================

print("\nSaving DPO adapter...")

trainer.save_model(OUTPUT_DIR)

print(f"DPO adapter saved to: {OUTPUT_DIR}")