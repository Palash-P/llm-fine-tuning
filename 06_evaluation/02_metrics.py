import json
import re


INPUT_FILE = "06_evaluation/evaluation_outputs.json"


# ============================================================
# LOAD RESULTS
# ============================================================

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    all_results = json.load(f)


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def calculate_format_score(response):
    """
    Measures whether the response follows our target
    Definition -> Example -> Key Point structure.
    """

    response_lower = response.lower()

    definition = "definition:" in response_lower
    example = "example:" in response_lower
    key_point = (
        "key point:" in response_lower
        or "key points:" in response_lower
    )

    score = sum([
        definition,
        example,
        key_point,
    ])

    return score / 3


def calculate_length(response):
    """
    Number of whitespace-separated words.
    """

    return len(response.split())


def contains_hallucination(response, prompt):
    """
    Small rule-based check for known incorrect RAG claims.

    This is intentionally NOT a general hallucination detector.
    It only checks a known failure mode in our experiment.
    """

    if "rag" not in prompt.lower():
        return False

    incorrect_terms = [
        "reinforcement learning aggregator",
        "reinforcement learning agent",
        "rapid application generation",
    ]

    response_lower = response.lower()

    return any(
        term in response_lower
        for term in incorrect_terms
    )


# ============================================================
# CALCULATE METRICS
# ============================================================

summary = []


for model_result in all_results:

    model_name = model_result["model"]

    results = model_result["results"]

    format_scores = []
    lengths = []
    hallucinations = 0

    for item in results:

        prompt = item["prompt"]
        response = item["response"]

        format_scores.append(
            calculate_format_score(response)
        )

        lengths.append(
            calculate_length(response)
        )

        if contains_hallucination(
            response,
            prompt,
        ):
            hallucinations += 1

    average_format_score = (
        sum(format_scores) / len(format_scores)
    )

    average_length = (
        sum(lengths) / len(lengths)
    )

    hallucination_rate = (
        hallucinations / len(results)
    )

    summary.append(
        {
            "model": model_name,
            "format_score": average_format_score,
            "average_response_words": average_length,
            "known_rag_hallucination_rate": hallucination_rate,
            "inference_time_seconds": model_result[
                "inference_time_seconds"
            ],
        }
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("=" * 80)
print("EVALUATION METRICS")
print("=" * 80)

print()

print(
    f"{'Model':<12}"
    f"{'Format':>12}"
    f"{'Words':>12}"
    f"{'RAG Error':>15}"
    f"{'Time (s)':>15}"
)

print("-" * 80)


for result in summary:

    print(
        f"{result['model']:<12}"
        f"{result['format_score'] * 100:>11.1f}%"
        f"{result['average_response_words']:>12.1f}"
        f"{result['known_rag_hallucination_rate'] * 100:>14.1f}%"
        f"{result['inference_time_seconds']:>15.2f}"
    )


print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

print(
    """
Format score:
Percentage of Definition / Example / Key Point sections
present in the response.

Known RAG hallucination rate:
Whether the model produced one of the known incorrect
RAG expansions observed during this experiment.

Important:
This is NOT a general hallucination detector.
It is a targeted metric for a known failure mode.

Response length:
Average number of generated words.

Inference time:
Total time required to generate all evaluation responses.
"""
)