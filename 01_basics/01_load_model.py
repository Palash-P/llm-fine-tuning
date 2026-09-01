from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto",
)

print("Model loaded!")
print(f"Device: {model.device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")


prompt = "Explain machine learning to a beginner in three sentences."


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
        max_new_tokens=100,
    )


response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True,
)


print("\n--- RESPONSE ---")
print(response)