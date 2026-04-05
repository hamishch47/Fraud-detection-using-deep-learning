"""FastAPI scoring service for the fraud detection pipeline.

Run with:
    uvicorn scoring_service:app --host 0.0.0.0 --port 8000

The service tries to load ``stacked_hybrid.pkl`` from the repository root.
If that artifact is not present it falls back to deterministic rule-based
scoring so the dashboard can be demoed without a trained model.

Replacing the fallback with your trained stacked hybrid model is
straightforward — see the ``_load_model`` / ``_score_with_model`` section.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_MODEL_PATH = Path(os.getenv("MODEL_PATH", "stacked_hybrid.pkl"))
_model: Any = None


def _load_model() -> Any:
    """Attempt to load the stacked hybrid model artifact.

    Returns the model object, or None when the artifact is absent.
    Separate from startup so the service starts even without artifacts.
    """
    if not _MODEL_PATH.exists():
        return None
    try:
        import joblib  # noqa: PLC0415

        return joblib.load(_MODEL_PATH)
    except Exception:
        return None


_model = _load_model()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Fraud Detection Scoring Service",
    description=(
        "POST /score to obtain a risk score, decision, and reason codes "
        "for a transaction payload."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class TransactionRequest(BaseModel):
    id: str
    amount: float
    merchant: str = ""
    location: str = ""
    velocity_1h: int = 0
    geo_distance_km: float = 0.0
    device_new: int = 0


class ScoreResponse(BaseModel):
    transaction_id: str
    risk_score: float  # 0–100
    decision: str  # "Pending Review" | "Auto-Approved"
    reason_codes: list[str]


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------

_HIGH_RISK_THRESHOLD = 75.0  # risk_score >= this → Pending Review


def _score_with_model(txn: TransactionRequest) -> tuple[float, list[str]]:
    """Run the stacked hybrid model and return (raw_probability, feature_flags).

    Replace the body of this function when wiring in new model artifacts.
    The function should return a probability in [0, 1].
    """
    # Example: build feature vector and run stacked pipeline.
    # import numpy as np
    # features = np.array([[txn.amount, txn.velocity_1h, txn.geo_distance_km, txn.device_new]])
    # prob = float(_model.predict_proba(features)[0, 1])
    # return prob, []
    raise NotImplementedError(
        "Replace this function body with your stacked hybrid model inference code. "
        "It should return a tuple of (probability: float in [0, 1], reason_codes: list[str]). "
        "See _fallback_score() below for the expected return format."
    )


def _fallback_score(txn: TransactionRequest) -> tuple[float, list[str]]:
    """Deterministic rule-based fallback when model artifact is missing.

    Returns a probability in [0, 1] and a list of triggered rule names.
    Rules are intentionally simple so the pipeline can be demoed end-to-end
    without a trained model.
    """
    score = 0.0
    reasons: list[str] = []

    # High-value transaction
    if txn.amount >= 50_000:
        score += 0.35
        reasons.append("high_amount")

    # Unusual velocity
    if txn.velocity_1h >= 5:
        score += 0.30
        reasons.append("high_velocity_1h")

    # Large geo distance from home
    if txn.geo_distance_km >= 500:
        score += 0.20
        reasons.append("geo_mismatch")

    # Transaction from a new device
    if txn.device_new:
        score += 0.15
        reasons.append("new_device")

    # Unknown / suspicious location keywords
    suspicious_locations = {"unknown", "moscow, ru", "proxy", "vpn"}
    if any(kw in txn.location.lower() for kw in suspicious_locations):
        score += 0.20
        reasons.append("suspicious_location")

    return min(score, 1.0), reasons


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/healthz", summary="Health check")
def healthz() -> dict:
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/score", response_model=ScoreResponse, summary="Score a transaction")
def score(txn: TransactionRequest) -> ScoreResponse:
    """Return a risk score and decision for the given transaction payload."""
    if _model is not None:
        try:
            prob, reasons = _score_with_model(txn)
        except Exception:
            # If model inference fails for any reason, fall back gracefully.
            prob, reasons = _fallback_score(txn)
    else:
        prob, reasons = _fallback_score(txn)

    risk_score = round(prob * 100, 2)
    decision = "Pending Review" if risk_score >= _HIGH_RISK_THRESHOLD else "Auto-Approved"

    return ScoreResponse(
        transaction_id=txn.id,
        risk_score=risk_score,
        decision=decision,
        reason_codes=reasons,
    )
