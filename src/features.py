"""Feature engineering shared by training, API and simulator."""

import numpy as np
import pandas as pd

# Original dataset column name -> short name
RENAME = {
    "Air temperature [K]": "air_temp",
    "Process temperature [K]": "process_temp",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
}

RAW_SIGNALS = ["air_temp", "process_temp", "rpm", "torque", "tool_wear"]

FEATURE_COLUMNS = RAW_SIGNALS + [
    "power_w", "temp_diff", "wear_torque", "type_M", "type_H",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Take raw readings (with a 'Type' column) and return model-ready features."""
    out = df.rename(columns=RENAME).copy()

    out["power_w"] = out["torque"] * out["rpm"] * 2 * np.pi / 60
    out["temp_diff"] = out["process_temp"] - out["air_temp"]
    out["wear_torque"] = out["tool_wear"] * out["torque"]

    out["type_M"] = (out["Type"] == "M").astype(int)
    out["type_H"] = (out["Type"] == "H").astype(int)

    return out[FEATURE_COLUMNS]
