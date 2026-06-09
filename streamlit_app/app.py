from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR / "src"))

import json

import joblib
import pandas as pd
import streamlit as st

from fraud_detection.config import METRICS_PATH, MODEL_PATH, THRESHOLD_PATH

st.set_page_config(page_title="Fraud Risk Scoring", layout="wide")
st.title("Fraud Risk Scoring")

if not MODEL_PATH.exists() or not THRESHOLD_PATH.exists() or not METRICS_PATH.exists():
    st.warning("Model artifacts are missing. Train the model before using the dashboard.")
    st.stop()

model = joblib.load(MODEL_PATH)
default_threshold = float(joblib.load(THRESHOLD_PATH))
metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
feature_order = metrics["features"]

uploaded_file = st.file_uploader("Upload transactions CSV", type=["csv"])
threshold = st.slider("Decision threshold", 0.01, 0.99, default_threshold, 0.01)

if uploaded_file:
    transactions = pd.read_csv(uploaded_file)
    missing = [feature for feature in feature_order if feature not in transactions.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    scores = model.predict_proba(transactions[feature_order])[:, 1]
    output = transactions.copy()
    output["fraud_probability"] = scores
    output["is_flagged"] = output["fraud_probability"] >= threshold
    output = output.sort_values("fraud_probability", ascending=False)

    flagged_count = int(output["is_flagged"].sum())
    col1, col2, col3 = st.columns(3)
    col1.metric("Transactions", len(output))
    col2.metric("Flagged", flagged_count)
    col3.metric("Threshold", f"{threshold:.2f}")

    st.dataframe(output.head(100), use_container_width=True)

