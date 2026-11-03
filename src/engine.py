"""One entry point: raw reading -> full analysis."""

from src.anomaly import detect
from src.explain import explain


def analyze(reading: dict, shown_risk=None) -> dict:
    result = explain(reading, shown_risk)   # risk, status, explanation, recommendation
    result["anomaly"] = detect(reading)
    result["reading"] = reading
    return result