# ============================================
# JOBSHIELD — LOCAL AI EXPLANATION LAYER
# ============================================

import requests

try:
    from src.predict import predict_job
except ModuleNotFoundError:
    from predict import predict_job


MODEL_NAME = "qwen3:1.7b"


# ============================================
# WEAK / GENERIC FEATURES
# ============================================

WEAK_FEATURES = {
    # Common stop words
    "a", "an", "and", "are", "as", "at", "be", "by",
    "for", "from", "in", "is", "it", "of", "on", "or",
    "that", "the", "this", "to", "was", "we", "were",
    "with", "our", "your", "you", "their", "they",
    "will", "can", "has", "have", "had", "do", "does",
    "did", "not", "no",

    # Very short / noisy tokens
    "en", "ll", "re", "us",

    # Generic words
    "about",
    "ago",
    "also",
    "back",
    "based",
    "been",
    "being",
    "below",
    "between",
    "both",
    "could",
    "days",
    "during",
    "each",
    "get",
    "getting",
    "here",
    "how",
    "into",
    "just",
    "more",
    "most",
    "much",
    "new",
    "non",
    "now",
    "other",
    "over",
    "same",
    "some",
    "such",
    "than",
    "then",
    "there",
    "these",
    "those",
    "through",
    "under",
    "very",
    "want",
    "while",

    # Generic phrases
    "want to",
    "for more",
    "more information",
    "apply now",
    "learn more",
}


# ============================================
# SELECT MEANINGFUL FEATURES
# ============================================

def select_meaningful_features(features, top_n=3):
    """
    Select useful model features while removing
    generic words and generic phrases.
    """

    selected = []

    for item in features:

        feature = item["feature"].lower().strip()

        # Remove exact weak words or phrases
        if feature in WEAK_FEATURES:
            continue

        words = feature.split()

        # Remove phrases made entirely from weak words
        if words and all(word in WEAK_FEATURES for word in words):
            continue

        selected.append(item)

        if len(selected) >= top_n:
            break

    return selected


# ============================================
# GET CLEAN USER-FACING SIGNALS
# ============================================

def get_display_signals(result, top_n=5):
    """
    Select the same cleaned evidence that should be
    shown to the user interface and supplied to Qwen.

    This keeps the explanation layer consistent with
    the model evidence.
    """

    fraud_features = select_meaningful_features(
        result["fraud_evidence"],
        top_n=top_n
    )

    legitimate_features = select_meaningful_features(
        result["legitimate_evidence"],
        top_n=top_n
    )

    if result["prediction"] == "Potentially Fraudulent":
        model_signals = fraud_features
        counter_signals = legitimate_features
    else:
        model_signals = legitimate_features
        counter_signals = fraud_features

    return {
        "model_signals": model_signals,
        "counter_signals": counter_signals
    }


# ============================================
# BUILD AI EXPLANATION PROMPT
# ============================================

def build_explanation_prompt(result):
    """
    Create a grounded, user-friendly explanation prompt.

    The ML model makes the classification.
    Python selects the evidence.
    Qwen converts the supplied evidence
    into simple language.
    """

    signals = get_display_signals(result, top_n=5)

    model_features = signals["model_signals"][:3]
    counter_features = signals["counter_signals"][:2]

    model_evidence = "\n".join(
        f"- {item['feature']}"
        for item in model_features
    ) or "- None"

    counter_evidence = "\n".join(
        f"- {item['feature']}"
        for item in counter_features
    ) or "- None"

    prompt = f"""
You are the user-facing explanation assistant for JobShield.

The machine-learning model has already made the classification.

You MUST NOT change the classification.

Your job is to explain the result in simple,
natural language that an ordinary job seeker
can easily understand.

============================================
MODEL RESULT
============================================

Prediction: {result["prediction"]}
Model score: {result["fraud_probability"]}
Risk level: {result["risk_level"]}

============================================
FEATURES SUPPORTING THE MAIN PREDICTION
============================================

{model_evidence}

============================================
FEATURES PULLING IN THE OPPOSITE DIRECTION
============================================

{counter_evidence}

============================================
IMPORTANT RULES
============================================

- Use ONLY the supplied features.
- Do not invent evidence.
- Do not introduce outside information.
- Do not add features that were not supplied.
- Do not change the prediction.
- Do not describe an individual word as proof of fraud.
- Do not describe an individual word as proof of legitimacy.
- Do not call an individual feature a scam tactic.
- Do not call an individual feature a red flag.
- Do not infer criminal intent.
- Do not infer deception.
- Do not infer financial motive.
- Do not infer wrongdoing.
- Do not say that the model score is the actual probability
  that the job is fraudulent.
- Do not describe the risk level as a real-world probability.
- Do not mention coefficients.
- Do not mention weights.
- Do not mention TF-IDF.
- Do not mention mathematical calculations.
- Do not mention statistical contribution values.
- Do not mention internal model reasoning.
- Keep the explanation short.
- Use simple language suitable for a general job seeker.

============================================
HOW TO EXPLAIN THE FEATURES
============================================

Explain the supplied words or phrases as patterns
the model learned from historical job postings.

Use simple language that a normal job seeker
can understand.

For example:

"The term 'data entry' was one of the patterns
the model associated more with potentially
fraudulent job postings in its training data."

IMPORTANT:

- Explain ONLY the supplied features.
- Do not replace them with different features.
- Do not say a feature proves anything.
- Do not call a feature a scam indicator or red flag.
- Explain that these are patterns learned from
  historical job postings.
- Keep each explanation to one short sentence.
- Avoid technical machine-learning terminology.
- Write for someone who is deciding whether to
  apply for a job, not for a machine-learning engineer.

============================================
REQUIRED OUTPUT
============================================

RETURN EXACTLY THIS FORMAT:

Summary:

Write 2 short, natural sentences.

- Start with the exact prediction.
- Give the model score.
- Explain that the model found patterns that supported
  this result.
- Do not describe the score as a real-world probability.

Why the model gave this result:

- Explain the first supplied feature in simple language.
- Explain the second supplied feature in simple language.
- Explain the third supplied feature in simple language.

What pulled the score the other way:

- Explain up to 2 supplied opposite-side features.
- Use simple language.
- Make clear that these features alone do not mean
  the job is fraudulent or legitimate.
- If there are no supplied opposite-side features, write:
  No strong signals were found on the opposite side.

STYLE:

- Sound helpful and professional.
- Use language that an ordinary job seeker understands.
- Avoid phrases such as "statistically contributed",
  "feature weight", "coefficient", "TF-IDF", or
  "classification boundary".
- Do not sound overly technical.
- Do not use complicated explanations.
- Keep each bullet to one short sentence.
- Do not add headings other than the required headings.
- Do not add a Caution section.
- Do not add extra commentary.
"""

    return prompt


# ============================================
# CALL LOCAL QWEN MODEL
# ============================================

def explain_prediction(result):
    """
    Send the ML result to the local Qwen model
    through the Ollama HTTP API.
    """

    prompt = build_explanation_prompt(result)

    try:

        response = requests.post(
            "http://127.0.0.1:11434/api/chat",

            json={
                "model": MODEL_NAME,
                "stream": False,
                "think": False,

                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                "options": {
                    "temperature": 0.2,
                    "num_predict": 300
                }
            },

            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        explanation = (
            data
            .get("message", {})
            .get("content", "")
            .strip()
        )

        if not explanation:
            raise ValueError(
                "Ollama returned an empty explanation."
            )

        # Remove any caution generated by Qwen
        if "Caution:" in explanation:
            explanation = explanation.split(
                "Caution:",
                1
            )[0].rstrip()

        # Standardized caution
        caution = (
            "\n\nCaution:\n"
            "The model prediction is not proof that the job "
            "is fraudulent or legitimate. "
            "Individual features do not establish fraud "
            "on their own."
        )

        explanation += caution

        return explanation

    except requests.RequestException as error:

        raise RuntimeError(
            f"Could not connect to Ollama: {error}"
        ) from error


# ============================================
# LOCAL TEST
# ============================================

if __name__ == "__main__":

    sample_job = """
    We are hiring a data entry assistant.
    Work from home and earn money quickly.
    Apply using the link below or call us
    for more information.
    """

    print("============================================")
    print("JOBSHIELD — ML PREDICTION")
    print("============================================")

    result = predict_job(sample_job)

    print("\nPrediction:", result["prediction"])
    print("Model score:", result["fraud_probability"])
    print("Risk level:", result["risk_level"])

    signals = get_display_signals(result)

    print("\nModel signals:")

    for item in signals["model_signals"]:
        print(
            f"  {item['feature']} "
            f"({item['contribution']:+.3f})"
        )

    print("\nCounter-signals:")

    for item in signals["counter_signals"]:
        print(
            f"  {item['feature']} "
            f"({item['contribution']:+.3f})"
        )

    print("\n============================================")
    print("JOBSHIELD — AI EXPLANATION")
    print("============================================")

    try:

        explanation = explain_prediction(result)

        print("\n" + explanation)

    except Exception as error:

        print("\nAI explanation failed.")
        print("Error:", error)