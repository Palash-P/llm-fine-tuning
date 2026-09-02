import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DPO_ADAPTER = "05_dpo/output_v1"

PROMPTS = [
    "Explain what an API is.",
    "Explain what Redis is.",
    "Explain what JWT authentication is.",
    "Explain what a vector database is.",
]


print("=" * 60)
print("DPO INFERENCE")
print("=" * 60)

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


print("\nCreating 4-bit quantization configuration...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)


print("\nLoading base model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)


print("\nLoading DPO adapter...")

model = PeftModel.from_pretrained(
    model,
    DPO_ADAPTER,
)

model.eval()

print("DPO adapter loaded.")


def generate_response(prompt):
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

    with torch.inference_mode():

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

    return response.strip()


print("\n" + "=" * 60)
print("GENERATING RESPONSES")
print("=" * 60)


for i, prompt in enumerate(PROMPTS, start=1):

    print(f"\n{'-' * 60}")
    print(f"PROMPT {i}")
    print(f"{'-' * 60}")

    print(f"\nUser:\n{prompt}")

    response = generate_response(prompt)

    print(f"\nDPO Model:\n{response}")


print("\n" + "=" * 60)
print("INFERENCE COMPLETE")
print("=" * 60)