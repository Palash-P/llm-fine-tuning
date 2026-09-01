import torch

from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
)


lora_config = LoraConfig(
    r=32,
    lora_alpha=16,
    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],

    bias="none",
    task_type="CAUSAL_LM",
)


model = get_peft_model(
    model,
    lora_config,
)


print("=" * 60)
print("TRAINABLE LoRA PARAMETERS")
print("=" * 60)

for name, parameter in model.named_parameters():

    if parameter.requires_grad:

        print(
            f"{name:80} "
            f"shape={str(tuple(parameter.shape)):20} "
            f"dtype={parameter.dtype}"
        )