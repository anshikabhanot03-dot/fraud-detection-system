from __future__ import annotations

import json
from enum import Enum
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fraud_detection.config import METRICS_PATH, MODEL_PATH, THRESHOLD_PATH

app = FastAPI(
    title="Real-Time Fraud Detection API",
    description="Scores transactions using a cost-sensitive fraud model.",
    version="0.1.0",
)


class RiskCategory(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class Transaction(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Transaction feature map, for example Time, Amount, V1...V28.",
    )


class PredictionResponse(BaseModel):
    fraud_probability: float
    threshold: float
    is_fraud: bool
    risk_category: RiskCategory


def load_artifacts() -> tuple[Any, float, list[str]]:
    if not MODEL_PATH.exists() or not THRESHOLD_PATH.exists() or not METRICS_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Model artifacts are missing. Run `python -m fraud_detection.train` first.",
        )
    model = joblib.load(MODEL_PATH)
    threshold = float(joblib.load(THRESHOLD_PATH))
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return model, threshold, metrics["features"]


def categorize_risk(probability: float, threshold: float) -> RiskCategory:
    if probability >= threshold:
        return RiskCategory.high
    if probability >= threshold * 0.5:
        return RiskCategory.medium
    return RiskCategory.low


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    model, threshold, feature_order = load_artifacts()
    missing = [feature for feature in feature_order if feature not in transaction.features]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required features: {missing}",
        )

    row = pd.DataFrame([{feature: transaction.features[feature] for feature in feature_order}])
    probability = float(model.predict_proba(row)[:, 1][0])
    return PredictionResponse(
        fraud_probability=probability,
        threshold=threshold,
        is_fraud=probability >= threshold,
        risk_category=categorize_risk(probability, threshold),
    )

