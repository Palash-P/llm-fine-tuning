from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

DATASET_PATH = "02_sft/dataset"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

dataset = load_dataset(
    "json",
    data_files={
        "train": f"{DATASET_PATH}/train.jsonl",
        "validation": f"{DATASET_PATH}/validation.jsonl",
    },
)

example = dataset["train"][0]

formatted_text = tokenizer.apply_chat_template(
    example["messages"],
    tokenize=False,
    add_generation_prompt=False,
)

print("=== ORIGINAL MESSAGES ===")

print(example["messages"])


print("\n=== FORMATTED TRAINING TEXT ===")

print(formatted_text)