"""Prediction engine: raw reading -> risk % and status. Prototype logic."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import engineer_features

ROOT = Path(__file__).resolve().parent.parent
_artifact = joblib.load(ROOT / "models" / "failure_model.joblib")

MODEL = _artifact["model"]
MODEL_NAME = _artifact["model_name"]
THRESHOLD = float(np.clip(_artifact["threshold"], 0.05, 0.95))

# Single-row predictions are faster without parallel workers
if hasattr(MODEL, "n_jobs"):
    MODEL.n_jobs = 1

# Prototype status bands on the 0-100 risk index (NOT safety standards)
WARNING_RISK = 30.0
CRITICAL_RISK = 50.0   # = the model's tuned failure threshold


def readings_to_features(readings: pd.DataFrame) -> pd.DataFrame:
    """Accept short names (air_temp...) plus 'type', or the original dataset format."""
    df = readings.copy()
    if "Type" not in df.columns:
        df["Type"] = df.pop("type") if "type" in df.columns else "L"
    return engineer_features(df)


def score_to_risk(score):
    """Map raw model score so that THRESHOLD -> 50%, 0 -> 0%, 1 -> 100%."""
    return np.interp(score, [0.0, THRESHOLD, 1.0], [0.0, 50.0, 100.0])


def status_from_risk(risk: float) -> str:
    if risk >= CRITICAL_RISK:
        return "CRITICAL"
    if risk >= WARNING_RISK:
        return "WARNING"
    return "HEALTHY"


def risk_for_frame(readings: pd.DataFrame) -> np.ndarray:
    """Risk % for many readings at once."""
    scores = MODEL.predict_proba(readings_to_features(readings))[:, 1]
    return score_to_risk(scores)


def predict_reading(reading: dict) -> dict:
    risk = float(risk_for_frame(pd.DataFrame([reading]))[0])
    return {"risk_pct": round(risk, 1), "status": status_from_risk(risk)}