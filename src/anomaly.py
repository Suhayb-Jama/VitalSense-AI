"""Anomaly detection: Isolation Forest trained on healthy rows only."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import engineer_features
from src.predict import readings_to_features

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "anomaly_model.joblib"
ANOMALY_FEATURES = ["air_temp", "process_temp", "rpm", "torque",
                    "tool_wear", "power_w", "temp_diff"]

_cache = {}


def train():
    df = pd.read_csv(ROOT / "ai4i+2020+predictive+maintenance+dataset/ai4i2020.csv" , encoding="utf-8-sig")
    healthy = df[df["Machine failure"] == 0]
    X = engineer_features(healthy)[ANOMALY_FEATURES]

    pipe = make_pipeline(
        StandardScaler(),
        IsolationForest(n_estimators=200, random_state=42, n_jobs=1),
    )
    pipe.fit(X)

    # Flag anything scoring lower than 99% of healthy training rows
    threshold = float(np.percentile(pipe.decision_function(X), 1))
    joblib.dump({"pipeline": pipe, "threshold": threshold}, MODEL_PATH)
    print(f"Trained on {len(X)} healthy rows. Threshold: {threshold:.4f}")
    print(f"Saved to {MODEL_PATH}")


def _load():
    if "art" not in _cache:
        _cache["art"] = joblib.load(MODEL_PATH)
    return _cache["art"]


def detect(reading: dict) -> dict:
    art = _load()
    X = readings_to_features(pd.DataFrame([reading]))[ANOMALY_FEATURES]
    d = float(art["pipeline"].decision_function(X)[0])
    return {"is_anomaly": bool(d < art["threshold"]), "score": round(-d, 3)}


if __name__ == "__main__":
    train()