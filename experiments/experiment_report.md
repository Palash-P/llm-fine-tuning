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