from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto",
)

messages = [
    {
        "role": "user",
        "content": "The capital of France is",
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(
    text,
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits

print("Logits shape:")
print(logits.shape)

next_token_logits = logits[:, -1, :]

top_k = 10

top_logits, top_token_ids = torch.topk(
    next_token_logits,
    top_k,
    dim=-1,
)

print("\nTop predictions:")

for logit, token_id in zip(
    top_logits[0],
    top_token_ids[0],
):
    token = tokenizer.decode([token_id.item()])

    print(
        f"{repr(token):20} "
        f"logit={logit.item():.4f}"
    )

probabilities = torch.softmax(
    next_token_logits,
    dim=-1,
)

top_probs, top_token_ids = torch.topk(
    probabilities,
    10,
    dim=-1,
)


print("\nTop predictions:")

for probability, token_id in zip(
    top_probs[0],
    top_token_ids[0],
):
    token = tokenizer.decode([token_id.item()])

    print(
        f"{repr(token):20} "
        f"probability={probability.item():.4%}"
    )