# LLM Fine-Tuning Experiments

## Experiment 1: Full SFT

### Objective

Teach Qwen2.5-0.5B-Instruct to answer technical questions
using a Definition → Example → Key Point format.

### Model

- Base model: Qwen/Qwen2.5-0.5B-Instruct
- Parameters: ~0.5B

### Dataset

- Training examples: 10
- Validation examples: 3
- Task: Technical question answering
- Target format: Definition → Example → Key Point

### Configuration

- Epochs: 3
- Batch size: 1
- Gradient accumulation: 4
- Learning rate: 2e-5
- Precision: BF16
- Maximum sequence length: 512

### Results

| Metric | Result |
|---|---:|
| Training time | 44.93 sec |
| Peak GPU memory | 4.50 GB |
| Final training loss | 1.469 |
| Final evaluation loss | 1.457 |
| Evaluation token accuracy | 68.22% |

### Qualitative Findings

The fine-tuned model consistently followed the
Definition → Example → Key Point format.

The behavior generalized beyond the training examples.
For example, Kubernetes was not present in the training
dataset, but the fine-tuned model still applied the learned
response structure.

### Tradeoffs

The fine-tuned model produced shorter and more structured
answers than the base model. In some cases, this came at the
cost of detail.

### Limitations

The dataset contains only 10 training examples and 3 validation
examples. Therefore, these results should not be considered
evidence of strong generalization.

### Conclusion

Full SFT successfully changed the model's response behavior
while preserving its ability to answer unseen technical
questions. This experiment provides the baseline for comparing
LoRA and QLoRA.

## Experiment 2: LoRA Setup

### Parameter Efficiency

Full fine-tuning:
- Total parameters: 494,032,768
- Trainable parameters: 494,032,768
- Trainable percentage: 100%

LoRA:
- Total parameters: 495,114,112
- Trainable parameters: 1,081,344
- Trainable percentage: 0.2184%

LoRA freezes the pretrained model and introduces
low-rank trainable adapters into the attention projection
layers.

Configuration:
- Rank (r): 8
- Alpha: 16
- Dropout: 0.05
- Target modules: q_proj, k_proj, v_proj, o_proj

### LoRA v1

Configuration:
- Rank: 8
- Alpha: 16
- Dropout: 0.05
- Target modules: q_proj, k_proj, v_proj, o_proj
- Learning rate: 2e-5

Results:
- Trainable parameters: 1,081,344
- Trainable percentage: 0.2184%
- Training time: 26.16 seconds
- Peak GPU memory: 1.75 GB
- Evaluation loss: 3.429
- Evaluation token accuracy: 42.88%

Finding:

LoRA achieved a substantial reduction in training memory
compared with full SFT, but at a learning rate of 2e-5 it
did not sufficiently adapt to the target response format.

### LoRA v2

Configuration:
- Rank: 8
- Alpha: 16
- Dropout: 0.05
- Target modules: q_proj, k_proj, v_proj, o_proj
- Learning rate: 1e-4

Results:
- Trainable parameters: 1,081,344
- Trainable percentage: 0.2184%
- Training time: 27.96 seconds
- Peak GPU memory: 1.75 GB
- Evaluation loss: 2.981
- Evaluation token accuracy: 48.05%

Finding:

Increasing the learning rate from 2e-5 to 1e-4 improved
evaluation loss and token accuracy, but qualitative
evaluation showed that the model still largely retained
the base model's response style rather than consistently
following the Definition → Example → Key Point format.

### LoRA v3

Configuration:

* Rank: 32
* Alpha: 16
* Dropout: 0.05
* Target modules: q_proj, k_proj, v_proj, o_proj
* Learning rate: 1e-4

Results:

* Trainable parameters: 4,325,376
* Trainable percentage: 0.8679%
* Training time: 27.92 seconds
* Peak GPU memory: 1.80 GB
* Evaluation loss: 2.985
* Evaluation token accuracy: 47.69%

Finding:

Increasing the rank from 8 to 32 while keeping alpha fixed at 16 did not improve performance compared with LoRA v2.

However, this was not a clean rank comparison because the LoRA scaling factor changed from alpha/r = 2 in v2 to alpha/r = 0.5 in v3.

Therefore, the result cannot be used to conclude that rank 32 is ineffective. A controlled experiment was required where the alpha/r ratio remained constant.

### LoRA v4 — Controlled Rank Experiment

Configuration:

* Rank: 32
* Alpha: 64
* Dropout: 0.05
* Target modules: q_proj, k_proj, v_proj, o_proj
* Learning rate: 1e-4
* Alpha/r: 2

Results:

* Trainable parameters: 4,325,376
* Trainable percentage: 0.8679%
* Training time: 22.26 seconds
* Peak GPU memory: 1.80 GB
* Evaluation loss: 2.270
* Evaluation token accuracy: 55.08%

Finding:

Increasing the LoRA rank from 8 to 32 while keeping alpha/r constant at 2 produced a substantial improvement.

Evaluation loss decreased from 2.981 in v2 to 2.270, while evaluation token accuracy increased from 48.05% to 55.08%.

This indicates that a higher-rank adapter provided greater capacity to represent the required parameter update.

However, qualitative inference still produced mostly generic responses rather than consistently following the desired Definition → Example → Key Point format.

This showed that improved validation metrics alone did not guarantee the desired behavioral adaptation.

### LoRA v5 — Attention + MLP Targeting

Configuration:

* Rank: 32
* Alpha: 64
* Dropout: 0.05
* Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
* Learning rate: 1e-4
* Alpha/r: 2

Results:

* Trainable parameters: 17,596,416
* Trainable percentage: 3.4393%
* Training time: 28.13 seconds
* Peak GPU memory: 2.03 GB
* Evaluation loss: 1.480
* Evaluation token accuracy: 68.62%

Finding:

Adding the MLP projection layers to the LoRA target modules produced the strongest LoRA result.

Compared with v4:

* Evaluation loss decreased from 2.270 to 1.480.
* Evaluation token accuracy increased from 55.08% to 68.62%.
* Peak GPU memory increased only from 1.80 GB to 2.03 GB.

Most importantly, qualitative inference showed that the model consistently adopted the intended response structure:

Definition → Example → Key Point

for API, Celery, and Kubernetes prompts.

This demonstrated that targeting both attention and MLP projections provided substantially stronger behavioral adaptation for this experiment than attention-only LoRA.

### Overall LoRA Findings

The experiments showed that LoRA performance depended not only on the number of trainable parameters but also on where the adapters were applied.

Increasing rank from 8 to 32 improved quantitative performance when the alpha/r scaling factor was controlled. However, the attention-only configuration still showed limited qualitative behavioral adaptation.

Adding LoRA adapters to the MLP projections produced a much stronger improvement in both validation metrics and generated behavior.

The final v5 configuration trained only 3.44% of the model's parameters while achieving an evaluation loss of 1.480 and 68.62% token accuracy.

The experiment also demonstrated why fine-tuning evaluation should not rely solely on loss or token accuracy. Qualitative inference was necessary to verify whether the model had actually learned the intended response behavior.

Because the dataset contained only 10 training examples and 3 validation examples, these findings should be considered specific to this experiment rather than universal conclusions about LoRA.


## Experiment 3: QLoRA

### Objective

Evaluate whether 4-bit quantization of the frozen base model can reduce GPU memory consumption while maintaining the behavioral adaptation achieved with LoRA.

### Configuration

* Model: `Qwen/Qwen2.5-0.5B-Instruct`
* Quantization: 4-bit
* Quantization type: NF4
* Double quantization: Enabled
* Compute dtype: BF16
* LoRA rank: 32
* LoRA alpha: 64
* LoRA dropout: 0.05
* Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
* Learning rate: 1e-4
* Epochs: 3
* Dataset: 10 training examples, 3 validation examples
* GPU: NVIDIA RTX 4060 Laptop GPU, 8 GB VRAM

### Quantization Inspection

The 4-bit model loaded successfully using bitsandbytes.

The configuration used:

* NF4 quantization
* Double quantization
* BF16 compute

GPU memory after loading the quantized model was approximately 0.43 GB, compared with approximately 0.94 GB when loading the non-quantized model in the previous experiments.

The quantized weight parameters were represented using bitsandbytes' quantized storage mechanism, while some parameters such as embeddings and biases remained in BF16.

### QLoRA Training Results

| Metric                    |         LoRA v5 |        QLoRA v1 |
| ------------------------- | --------------: | --------------: |
| Base model precision      |            BF16 |       4-bit NF4 |
| LoRA rank                 |              32 |              32 |
| LoRA alpha                |              64 |              64 |
| Target modules            | Attention + MLP | Attention + MLP |
| Trainable parameters      |          17.60M |         ~17.60M |
| Trainable percentage      |           3.44% |          ~3.44% |
| Peak GPU memory           |         2.03 GB |         1.37 GB |
| Training time             |       28.13 sec |       38.27 sec |
| Evaluation loss           |           1.480 |           1.706 |
| Evaluation token accuracy |          68.62% |          66.07% |

### Finding 1: Memory Efficiency

QLoRA reduced peak GPU memory from 2.03 GB to 1.37 GB compared with the equivalent LoRA configuration.

This represents approximately a 32.5% reduction in peak GPU memory.

The reduction comes from storing the frozen base model weights using 4-bit quantization instead of higher-precision representation.

### Finding 2: Model Quality

QLoRA produced slightly worse validation metrics than LoRA v5.

Evaluation loss increased from 1.480 to 1.706, while token accuracy decreased from 68.62% to 66.07%.

Therefore, quantization introduced a measurable quality difference in this experiment.

However, the difference was relatively small compared with the memory savings.

### Finding 3: Qualitative Behavior

Despite the difference in validation metrics, QLoRA successfully preserved the behavioral adaptation learned by LoRA.

For API, Celery, and Kubernetes prompts, the model consistently generated responses following the desired:

`Definition → Example → Key Point`

structure.

Therefore, 4-bit quantization did not prevent the adapter from learning or expressing the target response behavior.

### Finding 4: Training Speed

QLoRA required approximately 38.27 seconds compared with 28.13 seconds for LoRA v5.

The additional overhead is not surprising because quantized weights require additional quantization/dequantization handling during computation.

For this small 0.5B model, the memory savings are more significant than the absolute training-time benefit because the model already fits comfortably in the available 8 GB GPU.

### Key Takeaway

QLoRA provided a substantial memory reduction while maintaining the desired behavioral adaptation.

In this experiment, QLoRA reduced peak GPU memory by approximately 32.5% while increasing evaluation loss modestly and increasing training time.

The primary advantage of QLoRA becomes more significant as model size increases, where loading and fine-tuning the full-precision base model can exceed available GPU memory.

### Important Lesson

QLoRA does not train the 4-bit base model directly.

Instead:

1. The pretrained base model is loaded using 4-bit quantization.
2. The quantized base model remains frozen.
3. Trainable LoRA adapters are added.
4. Computation uses BF16.
5. Only the LoRA parameters are updated during training.

This allows parameter-efficient fine-tuning with substantially lower memory requirements than using an unquantized base model.

The experiment also demonstrates that storage precision, compute precision, and trainable parameter precision are separate concepts.

## Experiment 4: DPO

### Objective

Evaluate whether Direct Preference Optimization (DPO) can teach the model to prefer higher-quality, structured responses over plausible but less useful responses.

### Setup

* **Base model:** Qwen/Qwen2.5-0.5B-Instruct
* **Base model quantization:** 4-bit NF4
* **Quantization:** Double quantization enabled
* **Compute dtype:** BF16
* **Adaptation:** LoRA
* **LoRA rank:** 32
* **LoRA alpha:** 64
* **Target modules:** q/k/v/o + gate/up/down
* **Learning rate:** 1e-5
* **Beta:** 0.1
* **Epochs:** 3
* **Training examples:** 10
* **Validation examples:** 3
* **Hardware:** NVIDIA RTX 4060 Laptop GPU, 8 GB VRAM

### Results

| Metric                   |    DPO v1 |
| ------------------------ | --------: |
| Training time            | 56.57 sec |
| Peak GPU memory          |   1.09 GB |
| Final train loss         |    0.5375 |
| Eval loss                |    0.5061 |
| Eval preference accuracy |      100% |
| Eval reward margin       |    0.4191 |

### Findings

DPO successfully learned the preference signal in the dataset. Preference accuracy reached **100% on the 3-example held-out validation set**, while the reward margin became positive, indicating that the policy assigned a higher relative preference to the chosen responses than the rejected responses.

However, qualitative inference showed that preference optimization did **not** guarantee factual correctness or consistently improved responses. The model continued to produce factual hallucinations, incomplete explanations, and inconsistent adherence to the desired `Definition → Example → Key Point` structure.

### Key Takeaway

DPO successfully optimized for the provided preference signal, but **preference accuracy should not be treated as a direct measure of factual correctness or overall response quality**. On this small dataset and 0.5B model, DPO improved the learned preference relationship without reliably eliminating hallucinations.

### Limitation

The experiment used only 10 training examples and 3 validation examples with a 0.5B parameter model. Therefore, the 100% validation preference accuracy should not be interpreted as evidence of broad alignment or generalization.

A larger preference dataset and a more capable base model would be required for stronger conclusions.


## Final Evaluation

### Evaluation Objective

The final evaluation compares all trained variants using the same evaluation prompts and generation settings.

The goal is to determine whether the training methods actually improved the model's downstream behavior rather than relying only on training or validation loss.

The following variants were evaluated:

* Base model
* Full SFT
* LoRA v5
* QLoRA v1
* DPO v1

### Evaluation Methodology

All five model variants were evaluated on the same 8 technical prompts:

1. API
2. Redis
3. Docker
4. RAG
5. AI Agent
6. JWT Authentication
7. Celery
8. Vector Database

Generation was performed using the same deterministic generation configuration:

* `do_sample=False`
* `max_new_tokens=150`

Two primary evaluation metrics were used.

#### 1. Format Adherence

Measures whether the generated response follows the target:

`Definition → Example → Key Point`

structure.

#### 2. Core Concept Coverage

Measures whether predefined core technical concepts expected for each question appear in the generated response.

This is a transparent concept-coverage metric rather than a complete factuality evaluator.

A targeted RAG hallucination diagnostic was also recorded. Since only one of the eight evaluation prompts tested RAG, this metric is treated as a targeted diagnostic rather than a general hallucination rate.

### Results

| Model    | Format Adherence | Core Concept Coverage | Avg. Response Words | Inference Time |
| -------- | ---------------: | --------------------: | ------------------: | -------------: |
| Base     |             0.0% |                 58.3% |               123.9 |         86.48s |
| SFT      |            62.5% |                 75.0% |                75.6 |         42.77s |
| LoRA v5  |            37.5% |                 75.0% |                94.5 |        144.62s |
| QLoRA v1 |            95.8% |                 87.5% |                76.1 |        113.63s |
| DPO v1   |             0.0% |                 70.8% |               125.5 |        189.05s |

### Finding 1: QLoRA Produced the Strongest Evaluation Result

QLoRA achieved the highest combination of the two primary evaluation metrics.

It achieved:

* **95.8% format adherence**
* **87.5% core concept coverage**

This was the strongest result among the evaluated variants.

Importantly, this was achieved while using only approximately **3.44% trainable parameters** and **1.37 GB peak GPU memory** during training.

The result suggests that QLoRA provided an effective balance between parameter efficiency, memory usage, and behavioral adaptation for this particular experiment.

### Finding 2: SFT Successfully Learned the Target Format

The base model achieved **0% format adherence**, while SFT achieved **62.5%**.

This demonstrates that supervised fine-tuning was effective at teaching the model the desired response structure.

SFT also increased core concept coverage from **58.3% to 75.0%**.

Therefore, the supervised examples influenced both response structure and technical concept coverage.

### Finding 3: LoRA Was Parameter Efficient but Required the Right Configuration

LoRA v5 achieved **75.0% core concept coverage** and **37.5% format adherence**.

The LoRA experiments demonstrated that adapter configuration had a significant effect on behavior.

In particular, targeting both attention and MLP projection layers produced substantially better results than attention-only targeting.

This reinforces an important practical lesson:

> Parameter-efficient fine-tuning does not automatically mean behavior-efficient fine-tuning.

The adapter placement, rank, scaling factor, learning rate, and training data all influence the final result.

### Finding 4: DPO Preference Accuracy Did Not Predict Generation Quality

DPO achieved:

* **100% preference accuracy** on the 3-example held-out validation set
* Positive reward margin of **0.4191**

However, the independent generation evaluation produced:

* **0% format adherence**
* **70.8% core concept coverage**

The model also continued to produce factual errors and inconsistent responses.

This demonstrates that optimizing a preference objective does not guarantee improvements on every downstream quality metric.

Preference accuracy and generation quality therefore need to be evaluated separately.

### Finding 5: Fine-Tuning Did Not Reliably Eliminate Hallucinations

The evaluation included a targeted RAG diagnostic.

The base model incorrectly interpreted RAG in the evaluation set, and the fine-tuned variants continued to produce incorrect expansions or explanations.

The observed RAG error rate was **12.5%**, corresponding to one incorrect RAG response out of eight evaluation prompts.

Because only one prompt tested RAG, this should **not** be interpreted as a general hallucination rate.

The result instead demonstrates that a very small fine-tuning dataset is insufficient to reliably correct factual knowledge in the model.

### Finding 6: Training Metrics Alone Were Not Enough

Several experiments demonstrated a gap between quantitative training metrics and actual generated behavior.

For example:

* LoRA v4 improved evaluation loss and token accuracy but still produced mostly generic responses.
* DPO achieved 100% preference accuracy but failed to consistently follow the target response format.
* QLoRA produced slightly worse validation loss than LoRA v5 but achieved stronger format adherence and concept coverage during independent evaluation.

Therefore, evaluation should include both:

**Training metrics → Loss, token accuracy, preference accuracy**

and

**Behavioral metrics → Format adherence, concept coverage, qualitative correctness**

### Overall Conclusion

The experiments demonstrate the practical tradeoffs between full SFT, LoRA, QLoRA, and DPO.

Full SFT was effective at teaching the target response structure.

LoRA significantly reduced the number of trainable parameters and demonstrated that adapter configuration strongly affects adaptation quality.

QLoRA provided the strongest combination of format adherence and core concept coverage in the final evaluation while substantially reducing memory consumption.

DPO successfully optimized the provided preference signal, but high preference accuracy did not translate into consistently better factual or structured responses.

The most important engineering lesson from these experiments is that **training metrics alone are insufficient to determine whether a fine-tuned model is actually better**.

A robust fine-tuning workflow should evaluate:

1. Training behavior
2. Validation performance
3. Parameter efficiency
4. Memory consumption
5. Generated response quality
6. Task-specific behavioral metrics
7. Failure cases and hallucinations

### Limitations

* Only 10 training examples and 3 validation examples were used.
* The DPO experiment used only 10 preference-training examples and 3 preference-validation examples.
* The base model contains only approximately 0.5B parameters.
* Final evaluation used only 8 prompts.
* Core concept coverage is not a complete factuality metric.
* Format adherence measures structure rather than answer quality.
* The RAG hallucination diagnostic is based on only one evaluation prompt.
* Inference-time measurements are affected by generated output length and the small evaluation size.
* Results are specific to this dataset, model, hardware, and configuration and should not be interpreted as universal conclusions about SFT, LoRA, QLoRA, or DPO.

### Final Engineering Takeaway

The complete experiment demonstrates a progression from:

`Full Fine-Tuning → LoRA → QLoRA → DPO → Independent Evaluation`

Each technique optimizes a different part of the fine-tuning problem.

The final results show why real-world LLM fine-tuning requires more than simply running a training script. The engineer must control the adaptation method, understand the memory/parameter tradeoffs, design appropriate evaluation metrics, inspect generated behavior, and analyze failure cases before deciding whether a model actually improved.
