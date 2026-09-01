from datasets import load_dataset

DATASET_PATH = "02_sft/dataset"

dataset = load_dataset(
    "json",
    data_files={
        "train": f"{DATASET_PATH}/train.jsonl",
        "validation": f"{DATASET_PATH}/validation.jsonl",
    },
)

print(dataset)

print("\nTraining example:")
print(dataset["train"][0])

print("\nValidation example:")
print(dataset["validation"][0])