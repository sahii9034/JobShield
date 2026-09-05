# ============================================
# JOBSHIELD — PREDICTION + EXPLAINABILITY
# ============================================

import joblib
import numpy as np
from pathlib import Path


# --------------------------------------------
# Load trained model artifacts
# --------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

tfidf = joblib.load(
    MODELS_DIR / "jobshield_tfidf.pkl"
)

model = joblib.load(
    MODELS_DIR / "jobshield_model.pkl"
)


# --------------------------------------------
# Decision threshold
# Selected using validation data
# --------------------------------------------

THRESHOLD = 0.60


# --------------------------------------------
# Common words we don't want to show
# as explanations
# --------------------------------------------

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by",
    "for", "from", "in", "is", "it", "of", "on", "or",
    "that", "the", "this", "to", "was", "we", "were",
    "with", "our", "your", "you", "their", "they",
    "will", "can", "has", "have", "had", "do", "does",
    "did", "not", "no"
}


# --------------------------------------------
# Main prediction function
# --------------------------------------------

def predict_job(job_text, top_n=10):
    """
    Predict whether a job posting is potentially
    fraudulent and return model-based evidence.
    """

    if not isinstance(job_text, str):
        raise TypeError("job_text must be a string.")

    if not job_text.strip():
        raise ValueError("Job text cannot be empty.")

    # ----------------------------------------
    # Transform job text
    # ----------------------------------------

    job_vector = tfidf.transform([job_text])

    # ----------------------------------------
    # Fraud probability
    # ----------------------------------------

    fraud_probability = model.predict_proba(
        job_vector
    )[0, 1]

    # ----------------------------------------
    # Prediction
    # ----------------------------------------

    is_fraud = fraud_probability >= THRESHOLD

    prediction = (
        "Potentially Fraudulent"
        if is_fraud
        else "Likely Legitimate"
    )

    # ----------------------------------------
    # Risk level
    # ----------------------------------------

    if fraud_probability >= 0.80:
        risk_level = "High"
    elif fraud_probability >= THRESHOLD:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # ----------------------------------------
    # Calculate feature contributions
    #
    # contribution =
    # TF-IDF value × model coefficient
    # ----------------------------------------

    feature_names = np.array(
        tfidf.get_feature_names_out()
    )

    coefficients = model.coef_[0]

    feature_values = job_vector.toarray().flatten()

    contributions = feature_values * coefficients

    # ----------------------------------------
    # Build evidence dataframe
    # ----------------------------------------

    evidence = []

    for feature, value, contribution in zip(
        feature_names,
        feature_values,
        contributions
    ):

        # Ignore features not present
        if value <= 0:
            continue

        # Ignore common/unhelpful words
        words = feature.lower().split()

        if all(
            word in STOP_WORDS
            for word in words
        ):
            continue

        evidence.append({
            "feature": feature,
            "contribution": float(contribution)
        })

    # ----------------------------------------
    # Sort evidence
    # ----------------------------------------

    fraud_evidence = sorted(
        [
            item for item in evidence
            if item["contribution"] > 0
        ],
        key=lambda x: x["contribution"],
        reverse=True
    )[:top_n]

    legitimate_evidence = sorted(
        [
            item for item in evidence
            if item["contribution"] < 0
        ],
        key=lambda x: x["contribution"]
    )[:top_n]

    # Round contributions
    for item in fraud_evidence:
        item["contribution"] = round(
            item["contribution"], 3
        )

    for item in legitimate_evidence:
        item["contribution"] = round(
            item["contribution"], 3
        )

    # ----------------------------------------
    # Return complete result
    # ----------------------------------------

        # Determine which evidence supports the final prediction.
    if prediction == "Potentially Fraudulent":
        model_signals = fraud_evidence
        counter_signals = legitimate_evidence
    else:
        model_signals = legitimate_evidence
        counter_signals = fraud_evidence

    return {
        "prediction": prediction,
        "fraud_probability": round(float(fraud_probability), 3),
        "risk_level": risk_level,
        "threshold": THRESHOLD,

        # Existing evidence
        "fraud_evidence": fraud_evidence,
        "legitimate_evidence": legitimate_evidence,

        # Evidence selected according to the prediction
        "model_signals": model_signals,
        "counter_signals": counter_signals
    }


# ============================================
# Local test
# ============================================

if __name__ == "__main__":

    sample_job = """
    We are hiring a data entry assistant.
    Work from home and earn money quickly.
    Apply using the link below or call us
    for more information.
    """

    result = predict_job(sample_job)

    print("============================================")
    print("JOBSHIELD PREDICTION")
    print("============================================")

    print(
        "\nPrediction:",
        result["prediction"]
    )

    print(
        "Fraud probability:",
        result["fraud_probability"]
    )

    print(
        "Risk level:",
        result["risk_level"]
    )

    print(
        "Threshold:",
        result["threshold"]
    )

    print("\n🔴 Fraud evidence:")

    for item in result["fraud_evidence"]:
        print(
            f"   {item['feature']} "
            f"({item['contribution']:+.3f})"
        )

    print("\n🟢 Legitimate evidence:")

    for item in result["legitimate_evidence"]:
        print(
            f"   {item['feature']} "
            f"({item['contribution']:+.3f})"
        )