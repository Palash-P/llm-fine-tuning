import json


INPUT_FILE = "06_evaluation/evaluation_outputs.json"
OUTPUT_FILE = "06_evaluation/correctness_results.json"


# ============================================================
# REFERENCE CONCEPTS
# ============================================================
#
# These are the minimum concepts we expect a technically
# correct answer to mention.
#
# This is a transparent, rule-based metric.
# It is NOT a general hallucination detector.
# ============================================================

REFERENCE_CONCEPTS = {

    "Explain what an API is.": [
        "application programming interface",
        "software applications",
        "communicate",
    ],

    "Explain what Redis is.": [
        "in-memory",
        "key-value",
        "data store",
    ],

    "Explain what Docker is.": [
        "container",
        "applications",
        "dependencies",
    ],

    "Explain what RAG is.": [
        "retrieval",
        "augmented",
        "generation",
    ],

    "Explain what an AI agent is.": [
        "agent",
        "goal",
        "actions",
    ],

    "Explain what JWT authentication is.": [
        "json web token",
        "authentication",
        "token",
    ],

    "Explain what Celery is.": [
        "task queue",
        "asynchronous",
        "tasks",
    ],

    "Explain what a vector database is.": [
        "vectors",
        "similarity",
        "search",
    ],
}


# ============================================================
# LOAD EVALUATION OUTPUTS
# ============================================================

print("=" * 80)
print("CORRECTNESS EVALUATION")
print("=" * 80)

print("\nLoading evaluation outputs...")

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8",
) as f:
    all_results = json.load(f)


# ============================================================
# SCORE ONE RESPONSE
# ============================================================

def calculate_concept_score(prompt, response):

    concepts = REFERENCE_CONCEPTS[prompt]

    response_lower = response.lower()

    matched = []

    for concept in concepts:

        if concept.lower() in response_lower:
            matched.append(concept)

    score = len(matched) / len(concepts)

    return score, matched


# ============================================================
# EVALUATE ALL MODELS
# ============================================================

all_correctness_results = []


for model_result in all_results:

    model_name = model_result["model"]

    print("\n" + "=" * 80)
    print(f"EVALUATING: {model_name}")
    print("=" * 80)

    model_scores = []

    for item in model_result["results"]:

        prompt = item["prompt"]
        response = item["response"]

        score, matched = calculate_concept_score(
            prompt,
            response,
        )

        model_scores.append(score)

        print(
            f"\n{prompt}"
        )

        print(
            f"Concept score: {score * 100:.1f}%"
        )

        print(
            f"Matched: {matched}"
        )

    average_score = (
        sum(model_scores)
        / len(model_scores)
    )

    all_correctness_results.append(
        {
            "model": model_name,
            "average_concept_score": average_score,
            "individual_scores": model_scores,
        }
    )


# ============================================================
# DISPLAY SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("CORRECTNESS SUMMARY")
print("=" * 80)

print()

print(
    f"{'Model':<15}"
    f"{'Concept Score':>20}"
)

print("-" * 40)


for result in all_correctness_results:

    print(
        f"{result['model']:<15}"
        f"{result['average_concept_score'] * 100:>18.1f}%"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        all_correctness_results,
        f,
        indent=2,
        ensure_ascii=False,
    )


print("\n" + "=" * 80)
print("CORRECTNESS EVALUATION COMPLETE")
print("=" * 80)

print(
    f"\nResults saved to: {OUTPUT_FILE}"
)

print(
    """
Important limitation:

This metric measures whether predefined core concepts
appear in the generated response.

It does NOT prove that the entire response is factually
correct.

A model can mention the correct concepts while still
containing incorrect or hallucinated information.
"""
)