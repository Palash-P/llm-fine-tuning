# LLM Fine-Tuning Lab

A hands-on experiment-driven project exploring **Supervised Fine-Tuning (SFT), LoRA, QLoRA, and Direct Preference Optimization (DPO)** using Qwen2.5-0.5B-Instruct.

The goal of this project is not simply to fine-tune a model, but to understand **how different adaptation techniques affect memory usage, trainable parameters, model behavior, and evaluation performance**.

---

## What This Project Covers

* Transformer inference fundamentals
* Tokenization and chat templates
* Next-token prediction
* Cross-entropy loss
* Supervised Fine-Tuning (SFT)
* Low-Rank Adaptation (LoRA)
* LoRA rank and scaling experiments
* Attention vs MLP adapter targeting
* 4-bit quantization
* NF4 quantization
* Double quantization
* QLoRA
* Direct Preference Optimization (DPO)
* Preference datasets
* Behavioral evaluation
* Parameter efficiency
* GPU memory analysis
* Failure analysis

---

## Model

**Base model:** `Qwen/Qwen2.5-0.5B-Instruct`

The model was selected because it is small enough to run experiments locally on an RTX 4060 Laptop GPU while still providing a realistic environment for studying modern LLM fine-tuning techniques.

---

## Hardware

| Component | Specification              |
| --------- | -------------------------- |
| GPU       | NVIDIA RTX 4060 Laptop GPU |
| VRAM      | 8 GB                       |
| Python    | 3.11.9                     |
| PyTorch   | 2.11.0+cu128               |
| CUDA      | 12.8                       |

---

## Repository Structure

```text
llm-fine-tuning/
│
├── 01_basics/
│   ├── 01_load_model.py
│   ├── 02_tokenizer.py
│   └── 03_next_token_prediction.py
│
├── 02_sft/
│   ├── dataset/
│   │   ├── train.jsonl
│   │   └── validation.jsonl
│   ├── 01_load_dataset.py
│   ├── 02_format_dataset.py
│   ├── 03_train.py
│   └── 04_inference.py
│
├── 03_lora/
│   ├── 01_lora_setup.py
│   ├── 02_train.py
│   ├── 03_inference.py
│   └── 04_inspect_adapter.py
│
├── 04_qlora/
│   ├── 01_quantization.py
│   ├── 02_train.py
│   └── 03_inference.py
│
├── 05_dpo/
│   ├── dataset/
│   │   ├── train.jsonl
│   │   └── validation.jsonl
│   ├── 01_load_dataset.py
│   ├── 02_train.py
│   └── 03_inference.py
│
├── 06_evaluation/
│   ├── 01_evaluate.py
│   ├── 02_metrics.py
│   ├── 03_correctness.py
│   └── evaluation_outputs.json
│
├── experiments/
│   └── experiment_report.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Experiments

## 1. Full SFT

The first experiment established the baseline fine-tuning approach.

The model was trained on 10 technical question-answer examples designed around:

`Definition → Example → Key Point`

### Result

| Metric               |    Result |
| -------------------- | --------: |
| Trainable parameters |   494.03M |
| Trainable percentage |      100% |
| Peak VRAM            |   4.50 GB |
| Training time        | 44.93 sec |
| Eval loss            |     1.457 |
| Eval token accuracy  |    68.22% |

SFT successfully taught the model the target response structure.

---

## 2. LoRA

LoRA was used to investigate parameter-efficient fine-tuning.

Instead of updating the entire model, trainable low-rank adapters were inserted into selected transformer layers.

### LoRA v5 Result

| Metric               |    Result |
| -------------------- | --------: |
| Trainable parameters |    17.60M |
| Trainable percentage |     3.44% |
| Peak VRAM            |   2.03 GB |
| Training time        | 28.13 sec |
| Eval loss            |     1.480 |
| Eval token accuracy  |    68.62% |

The experiment showed that adapter placement mattered significantly.

The final configuration targeted both:

* Attention projections
* MLP projections

This produced substantially stronger adaptation than attention-only targeting.

---

## 3. QLoRA

QLoRA combined:

* 4-bit NF4 quantization
* Double quantization
* BF16 computation
* LoRA adapters

### Result

| Metric               |   LoRA v5 |  QLoRA v1 |
| -------------------- | --------: | --------: |
| Base precision       |      BF16 | 4-bit NF4 |
| Trainable parameters |    17.60M |   ~17.60M |
| Trainable percentage |     3.44% |    ~3.44% |
| Peak VRAM            |   2.03 GB |   1.37 GB |
| Training time        | 28.13 sec | 38.27 sec |
| Eval loss            |     1.480 |     1.706 |
| Eval accuracy        |    68.62% |    66.07% |

QLoRA reduced peak GPU memory by approximately **32.5%**.

The tradeoff was slightly higher training time and slightly worse validation metrics in this small-scale experiment.

---

## 4. DPO

DPO was used to investigate preference optimization.

The model was trained using:

`Prompt → Chosen Response + Rejected Response`

### Result

| Metric              |    Result |
| ------------------- | --------: |
| Peak VRAM           |   1.09 GB |
| Training time       | 56.57 sec |
| Eval loss           |    0.5061 |
| Preference accuracy |      100% |
| Reward margin       |    0.4191 |

DPO successfully learned the preference signal in the tiny validation set.

However, independent generation evaluation showed that preference accuracy did not guarantee better factual or structured responses.

---

# Final Evaluation

All five variants were evaluated on the same 8 technical prompts.

### Results

| Model    | Format Adherence | Core Concept Coverage |
| -------- | ---------------: | --------------------: |
| Base     |             0.0% |                 58.3% |
| SFT      |            62.5% |                 75.0% |
| LoRA v5  |            37.5% |                 75.0% |
| QLoRA v1 |        **95.8%** |             **87.5%** |
| DPO v1   |             0.0% |                 70.8% |

### Key Findings

**QLoRA achieved the strongest combination of format adherence and concept coverage in this experiment.**

SFT demonstrated that supervised examples can effectively teach response structure.

LoRA demonstrated that parameter-efficient adaptation depends heavily on adapter configuration.

QLoRA demonstrated that substantial memory savings can be achieved while preserving strong behavioral adaptation.

DPO demonstrated an important distinction between **optimizing a preference objective and improving actual downstream generation quality**.

---

# Key Engineering Lessons

### 1. Loss is not enough

A lower validation loss does not automatically mean that the model learned the desired behavior.

Generated responses must be inspected.

### 2. Parameter efficiency matters

LoRA reduced the number of trainable parameters from hundreds of millions to millions while requiring significantly less GPU memory.

### 3. Adapter placement matters

Targeting only attention projections produced weaker adaptation than targeting both attention and MLP projections in this experiment.

### 4. Quantization changes the memory/quality tradeoff

QLoRA substantially reduced memory consumption but introduced a small quality and training-time tradeoff.

### 5. Preference accuracy is not factuality

DPO achieved 100% preference accuracy on the tiny validation set, but this did not eliminate factual errors.

### 6. Evaluation must be task-specific

Different objectives require different measurements:

* SFT → validation loss + behavioral adherence
* LoRA → parameter efficiency + adaptation quality
* QLoRA → memory efficiency + quality
* DPO → preference accuracy + downstream generation quality

---

# Limitations

This project is intentionally a small-scale local experiment.

* Training dataset contains only 10 examples.
* Validation datasets contain only 3 examples.
* DPO preference validation contains only 3 examples.
* Base model has approximately 0.5B parameters.
* Final evaluation contains only 8 prompts.
* Core concept coverage is not a complete factuality evaluator.
* Format adherence measures structure rather than answer quality.
* Inference-time measurements are noisy on such a small evaluation set.
* Results should not be generalized to larger models or datasets without further experimentation.

---

# Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* TRL
* PEFT
* bitsandbytes
* CUDA
* Qwen2.5-0.5B-Instruct

---

# Running the Project

Create and activate the virtual environment:

```bash
python -m venv .venv
```

Activate on Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the experiments in order:

```bash
python 01_basics/01_load_model.py
python 01_basics/02_tokenizer.py
python 01_basics/03_next_token_prediction.py
```

Then proceed through:

```text
02_sft/
03_lora/
04_qlora/
05_dpo/
06_evaluation/
```

---

# Experiment Philosophy

This repository follows an **experiment-first approach**.

Instead of asking only:

> "How do I fine-tune an LLM?"

the experiments investigate:

> "What changes when I fine-tune an LLM using different adaptation strategies?"

Each experiment records:

* Configuration
* Trainable parameters
* GPU memory
* Training time
* Validation metrics
* Generated behavior
* Failure cases
* Tradeoffs
* Conclusions

The detailed experiment history is available in:

`experiments/experiment_report.md`

---

# Final Takeaway

This project demonstrates the complete progression from basic transformer inference to modern parameter-efficient fine-tuning:

```text
Transformer Basics
       ↓
      SFT
       ↓
     LoRA
       ↓
    QLoRA
       ↓
      DPO
       ↓
Independent Evaluation
       ↓
Failure Analysis
```

The main lesson is that **LLM fine-tuning is not just a training problem**.

It is an engineering problem involving:

**model adaptation + memory efficiency + parameter efficiency + evaluation + failure analysis.**
