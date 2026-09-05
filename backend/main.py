from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.predict import predict_job
from src.explainer import explain_prediction, get_display_signals
from src.url_extractor import extract_job_text


# ============================================
# JOBSHIELD — FASTAPI BACKEND
# ============================================

app = FastAPI(
    title="JobShield API",
    description="AI-powered job scam detection API",
    version="1.0.0"
)


# ============================================
# CORS
# ============================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# REQUEST MODEL
# ============================================

class JobRequest(BaseModel):
    job_text: str | None = None
    job_url: str | None = None


# ============================================
# HEALTH CHECK
# ============================================

@app.get("/")
def root():
    return {
        "message": "JobShield API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ============================================
# JOB ANALYSIS
# ============================================

@app.post("/analyze")
def analyze_job(request: JobRequest):

    if not request.job_text and not request.job_url:
        raise HTTPException(
            status_code=400,
            detail="Provide either a job description or a job URL."
        )

    try:

        # ----------------------------------------
        # TEXT INPUT
        # ----------------------------------------

        if request.job_text and request.job_text.strip():
            job_text = request.job_text.strip()

        # ----------------------------------------
        # URL INPUT
        # ----------------------------------------

        elif request.job_url and request.job_url.strip():

            try:
                job_text = extract_job_text(
                    request.job_url
                )

            except ValueError as error:
                raise HTTPException(
                    status_code=400,
                    detail=str(error)
                )

        else:
            raise HTTPException(
                status_code=400,
                detail="The supplied job input is empty."
            )

        # ----------------------------------------
        # ML PREDICTION
        # ----------------------------------------

        result = predict_job(job_text)

        # ----------------------------------------
        # CLEAN USER-FACING SIGNALS
        # ----------------------------------------

        signals = get_display_signals(
            result,
            top_n=5
        )

        # ----------------------------------------
        # LOCAL AI EXPLANATION
        # ----------------------------------------

        try:

            explanation = explain_prediction(
                result
            )

        except Exception as error:

            print(
                "AI explanation error:",
                error
            )

            explanation = (
                "AI explanation is currently unavailable. "
                "The machine-learning prediction is still available."
            )

        # ----------------------------------------
        # RESPONSE
        # ----------------------------------------

        return {
            "prediction": result["prediction"],
            "fraud_probability": result["fraud_probability"],
            "risk_level": result["risk_level"],
            "threshold": result["threshold"],

            # Clean user-facing signals
            "model_signals": signals["model_signals"],
            "counter_signals": signals["counter_signals"],

            # Raw model evidence retained for debugging
            "fraud_evidence": result["fraud_evidence"],
            "legitimate_evidence": result["legitimate_evidence"],

            # AI explanation
            "explanation": explanation
        }

    except HTTPException:
        raise

    except Exception as error:

        print(
            "Analysis error:",
            error
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "An unexpected error occurred "
                "while analyzing the job."
            )
        )