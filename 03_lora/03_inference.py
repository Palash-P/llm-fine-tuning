import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from peft import PeftModel


BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

ADAPTER = "03_lora/output_v5"


print("=" * 60)
print("LOADING BASE MODEL")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    device_map="auto",
)

print("Base model loaded.")


print("\n" + "=" * 60)
print("LOADING LoRA ADAPTER")
print("=" * 60)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER,
)

print("LoRA adapter loaded.")


def generate_response(prompt):

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

    generated_tokens = outputs[
        0
    ][
        inputs["input_ids"].shape[1]:
    ]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )


prompts = [
    "What is an API?",
    "What is Celery?",
    "What is Kubernetes?",
]


for prompt in prompts:

    print("\n" + "=" * 60)
    print(f"PROMPT: {prompt}")
    print("=" * 60)

    response = generate_response(prompt)

    print(response)