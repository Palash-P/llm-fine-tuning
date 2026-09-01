# base vs fine-tuned inference comparisoncle
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
FINETUNED_MODEL = "02_sft/output"


def generate_response(model, tokenizer, prompt):
    messages = [
        {
            "role": "user",
            "content": prompt,
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
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
        )

    # Remove the input tokens so we only decode the new answer
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )


print("=" * 60)
print("LOADING BASE MODEL")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    device_map="auto",
)

print("Base model loaded.")


print("\n" + "=" * 60)
print("LOADING FINE-TUNED MODEL")
print("=" * 60)

finetuned_tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL)

finetuned_model = AutoModelForCausalLM.from_pretrained(
    FINETUNED_MODEL,
    dtype=torch.float16,
    device_map="auto",
)

print("Fine-tuned model loaded.")


prompts = [
    "What is an API?",
    "What is Celery?",
    "What is Kubernetes?",
]


for prompt in prompts:

    print("\n" + "=" * 60)
    print(f"PROMPT: {prompt}")
    print("=" * 60)

    print("\n--- BASE MODEL ---")

    base_response = generate_response(
        base_model,
        tokenizer,
        prompt,
    )

    print(base_response)

    print("\n--- FINE-TUNED MODEL ---")

    finetuned_response = generate_response(
        finetuned_model,
        finetuned_tokenizer,
        prompt,
    )

    print(finetuned_response)