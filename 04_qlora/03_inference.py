import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "04_qlora/output_v1"


# ============================================================
# QUANTIZATION CONFIG
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ============================================================
# BASE MODEL
# ============================================================

print("=" * 60)
print("LOADING 4-BIT BASE MODEL")
print("=" * 60)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

print("Base model loaded.")


# ============================================================
# LoRA ADAPTER
# ============================================================

print("\n" + "=" * 60)
print("LOADING QLoRA ADAPTER")
print("=" * 60)

model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH,
)

print("QLoRA adapter loaded.")


# ============================================================
# INFERENCE FUNCTION
# ============================================================

def generate(prompt):

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
        )

    input_length = inputs["input_ids"].shape[-1]

    generated_tokens = outputs[0][input_length:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return response


# ============================================================
# TEST PROMPTS
# ============================================================

prompts = [
    "What is an API?",
    "What is Celery?",
    "What is Kubernetes?",
]


for prompt in prompts:

    print("\n" + "=" * 60)
    print(f"PROMPT: {prompt}")
    print("=" * 60)

    response = generate(prompt)

    print(response)