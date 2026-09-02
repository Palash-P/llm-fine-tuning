import json
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

SFT_ADAPTER = "02_sft/output"
LORA_ADAPTER = "03_lora/output_v5"
QLORA_ADAPTER = "04_qlora/output_v1"
DPO_ADAPTER = "05_dpo/output_v1"

OUTPUT_FILE = "06_evaluation/evaluation_outputs.json"


PROMPTS = [
    "Explain what an API is.",
    "Explain what Redis is.",
    "Explain what Docker is.",
    "Explain what RAG is.",
    "Explain what an AI agent is.",
    "Explain what JWT authentication is.",
    "Explain what Celery is.",
    "Explain what a vector database is.",
]


# ============================================================
# HARDWARE
# ============================================================

print("=" * 60)
print("LLM FINE-TUNING EVALUATION")
print("=" * 60)

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Total VRAM: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )
else:
    raise RuntimeError("CUDA GPU not available.")


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# 4-BIT CONFIGURATION
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)


# ============================================================
# LOAD BASE MODEL
# ============================================================

def load_base_model():
    print("\nLoading base model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model.eval()

    return model


# ============================================================
# LOAD ADAPTER
# ============================================================

def load_full_model(model_path):
    print(f"Loading full model: {model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
    )

    model.eval()

    return model


def load_adapter(adapter_path):
    print(f"Loading adapter: {adapter_path}")

    model = load_base_model()

    model = PeftModel.from_pretrained(
        model,
        adapter_path,
    )

    model.eval()

    return model

# ============================================================
# GENERATION
# ============================================================

def generate_response(model, prompt):

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


# ============================================================
# EVALUATE ONE MODEL
# ============================================================

def evaluate_model(model_name, model):

    print("\n" + "=" * 60)
    print(f"EVALUATING: {model_name}")
    print("=" * 60)

    results = []

    start_time = time.time()

    for i, prompt in enumerate(PROMPTS, start=1):

        print(f"\n[{i}/{len(PROMPTS)}] {prompt}")

        response = generate_response(
            model,
            prompt,
        )

        print(response)

        results.append(
            {
                "prompt": prompt,
                "response": response,
            }
        )

    elapsed_time = time.time() - start_time

    print(
        f"\n{model_name} inference time: "
        f"{elapsed_time:.2f} seconds"
    )

    return {
        "model": model_name,
        "inference_time_seconds": elapsed_time,
        "results": results,
    }


# ============================================================
# RUN EVALUATION
# ============================================================

all_results = []


# ------------------------------------------------------------
# 1. BASE MODEL
# ------------------------------------------------------------

model = load_base_model()

all_results.append(
    evaluate_model(
        "Base",
        model,
    )
)

del model
torch.cuda.empty_cache()


# ------------------------------------------------------------
# 2. SFT
# ------------------------------------------------------------

model = load_full_model(SFT_ADAPTER)

all_results.append(
    evaluate_model(
        "SFT",
        model,
    )
)

del model
torch.cuda.empty_cache()


# ------------------------------------------------------------
# 3. LoRA
# ------------------------------------------------------------

model = load_adapter(LORA_ADAPTER)

all_results.append(
    evaluate_model(
        "LoRA v5",
        model,
    )
)

del model
torch.cuda.empty_cache()


# ------------------------------------------------------------
# 4. QLoRA
# ------------------------------------------------------------

model = load_adapter(QLORA_ADAPTER)

all_results.append(
    evaluate_model(
        "QLoRA v1",
        model,
    )
)

del model
torch.cuda.empty_cache()


# ------------------------------------------------------------
# 5. DPO
# ------------------------------------------------------------

model = load_adapter(DPO_ADAPTER)

all_results.append(
    evaluate_model(
        "DPO v1",
        model,
    )
)

del model
torch.cuda.empty_cache()


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        all_results,
        f,
        indent=2,
        ensure_ascii=False,
    )


print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)

print(f"\nResults saved to: {OUTPUT_FILE}")