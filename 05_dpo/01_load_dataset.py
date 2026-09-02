from datasets import load_dataset


TRAIN_FILE = "05_dpo/dataset/train.jsonl"
VALIDATION_FILE = "05_dpo/dataset/validation.jsonl"


print("=" * 60)
print("LOADING DPO DATASET")
print("=" * 60)


dataset = load_dataset(
    "json",
    data_files={
        "train": TRAIN_FILE,
        "validation": VALIDATION_FILE,
    },
)


print(dataset)


print("\n" + "=" * 60)
print("DATASET FEATURES")
print("=" * 60)

print(dataset["train"].features)


print("\n" + "=" * 60)
print("FIRST TRAINING EXAMPLE")
print("=" * 60)

example = dataset["train"][0]

print("\nPROMPT:")
print(example["prompt"])

print("\nCHOSEN:")
print(example["chosen"])

print("\nREJECTED:")
print(example["rejected"])