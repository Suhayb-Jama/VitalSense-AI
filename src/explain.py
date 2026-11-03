"""Explainability and maintenance recommendations.
All thresholds and texts are PROTOTYPE logic, not production safety standards."""

from pathlib import Path

import pandas as pd

from src.features import engineer_features
from src.predict import readings_to_features, risk_for_frame, status_from_risk

ROOT = Path(__file__).resolve().parent.parent
_raw = pd.read_csv(ROOT / "ai4i+2020+predictive+maintenance+dataset" / "ai4i2020.csv", encoding="utf-8-sig")
_healthy = engineer_features(_raw[_raw["Machine failure"] == 0])
BASE_MEAN, BASE_STD, BASE_MEDIAN = _healthy.mean(), _healthy.std(), _healthy.median()

HIGH_Z = 1.5            # +/-1.5 std from the healthy average -> HIGH / LOW
MIN_DRIVER_PTS = 3.0    # a signal is a "driver" if removing it drops risk by >= 3 points

GROUPS = {              # signal group -> raw columns replaced by healthy median
    "Temperature": ["air_temp", "process_temp"],
    "Cooling/fan speed": ["rpm"],
    "Load (torque)": ["torque"],
    "Equipment wear": ["tool_wear"],
}
SIGNAL_INFO = [         # (name shown, feature used to label it)
    ("Temperature", "process_temp"),
    ("Cooling/fan speed", "rpm"),
    ("Load (torque)", "torque"),
    ("Power (derived)", "power_w"),
    ("Equipment wear", "tool_wear"),
]
PHRASE = {
    "Temperature": "abnormal thermal conditions",
    "Cooling/fan speed": "abnormal cooling/fan speed",
    "Load (torque)": "elevated load",
    "Equipment wear": "advanced equipment wear",
}
ACTION = {
    "Temperature": "inspect cooling airflow and ambient conditions",
    "Cooling/fan speed": "check fan and cooling-system operation",
    "Load (torque)": "review and rebalance the workload on this unit",
    "Equipment wear": "schedule component inspection or replacement",
}
DISCLAIMER = ("Prototype logic: thresholds and recommendations are illustrative, "
              "not production safety standards.")


def _level(feature, value):
    z = (value - BASE_MEAN[feature]) / BASE_STD[feature]
    return "HIGH" if z >= HIGH_Z else "LOW" if z <= -HIGH_Z else "NORMAL"


def _display(feature, v):
    if feature == "process_temp":
        return f"{v - 273.15:.1f} °C"
    if feature == "rpm":
        return f"{v:.0f} rpm"
    if feature == "torque":
        return f"{v:.1f} Nm"
    if feature == "power_w":
        return f"{v:,.0f} W"
    return f"{v:.0f} min"


def _join(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def explain(reading: dict, shown_risk=None) -> dict:
    # One batch: the real reading + one variant per signal group
    variants = [dict(reading)]
    for cols in GROUPS.values():
        v = dict(reading)
        for c in cols:
            v[c] = float(BASE_MEDIAN[c])
        variants.append(v)
    risks = risk_for_frame(pd.DataFrame(variants))

    raw_risk = float(risks[0])
    risk = raw_risk if shown_risk is None else float(shown_risk)
    status = status_from_risk(risk)

    contributions = sorted(
        [{"signal": g, "impact": round(float(raw_risk - r), 1)} for g, r in zip(GROUPS, risks[1:])],
        key=lambda c: -c["impact"],
    )
    drivers = [c["signal"] for c in contributions if c["impact"] >= MIN_DRIVER_PTS]

    feats = readings_to_features(pd.DataFrame([reading])).iloc[0]
    signals = [{"name": n, "display": _display(f, float(feats[f])), "level": _level(f, float(feats[f]))}
               for n, f in SIGNAL_INFO]

    if status == "HEALTHY":
        cause = "All monitored signals are within the normal operating range."
        recommendation = "No action required. Continue routine monitoring."
    else:
        if drivers:
            cause = "Failure risk is driven by " + _join([PHRASE[d] for d in drivers[:3]]) + "."
            actions = [ACTION[d] for d in drivers[:2]]
        else:
            cause = "No single signal dominates, but the combination of readings looks unusual."
            actions = ["run a general inspection of the unit"]
        if status == "CRITICAL":
            recommendation = ("URGENT: " + _join(actions).capitalize() +
                              ". Schedule preventive maintenance immediately and consider moving workload to another unit.")
        else:
            recommendation = ("Plan soon: " + _join(actions).capitalize() +
                              ". Schedule a maintenance inspection in the next window.")

    return {
        "risk_pct": round(risk, 1),
        "status": status,
        "signals": signals,
        "contributions": contributions,
        "cause": cause,
        "recommendation": recommendation,
        "disclaimer": DISCLAIMER,
    }