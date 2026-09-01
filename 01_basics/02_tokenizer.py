from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

messages = [
    {
        "role": "user",
        "content": "Explain machine learning to a beginner.",
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print("=== FORMATTED TEXT ===")
print(text)

tokens = tokenizer.tokenize(text)

print("\n=== TOKENS ===")
print(tokens)

token_ids = tokenizer.encode(text)

print("\n=== TOKEN IDs ===")
print(token_ids)

print("\n=== NUMBER OF TOKENS ===")
print(len(token_ids))