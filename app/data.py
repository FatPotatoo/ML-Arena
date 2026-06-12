"""
Dataset loading and the fixed chronological 3-way split.

This module is the ONLY place that touches the raw CSV. Everything else in the
backend asks this module for data, so the split logic lives in exactly one spot.

The split is fixed (not a user choice, by design):
    train  = 2008-2014
    val    = 2015          (users see these metrics while tuning)
    test   = 2016+         (HIDDEN - the honest final score)
"""
from functools import lru_cache
from pathlib import Path

import pandas as pd

# weatherAUS.csv sits in  backend/data/  -> two folders up from this file (app/ -> backend/)
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "weatherAUS.csv"

TARGET = "RainTomorrow"

# The columns a user is allowed to pick as features, grouped by how we treat them.
NUMERIC_FEATURES = [
    "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine", "WindGustSpeed",
    "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am",
    "Pressure3pm", "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm",
]
CATEGORICAL_FEATURES = ["Location", "WindGustDir", "WindDir9am", "WindDir3pm"]
BINARY_FEATURES = ["RainToday"]  # Yes/No -> 1/0

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES


@lru_cache(maxsize=1)
def load_raw() -> pd.DataFrame:
    """Read the CSV once and cache it in memory.

    @lru_cache means the file is read from disk only the first time this is
    called; every later call returns the same DataFrame instantly.
    """
    df = pd.read_csv(DATA_PATH)
    # Rows with no label can't be trained or scored, so drop them up front.
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    return df


def split_chronological(df: pd.DataFrame):
    """Split by calendar year so the future never leaks into the past."""
    train = df[df["Year"] <= 2014].copy()
    val = df[df["Year"] == 2015].copy()
    test = df[df["Year"] >= 2016].copy()
    return train, val, test


def _years_label(part: pd.DataFrame) -> str:
    return f"{int(part['Year'].min())}-{int(part['Year'].max())}"


def dataset_info() -> dict:
    """A summary the frontend uses to build the feature-selection screen."""
    df = load_raw()
    train, val, test = split_chronological(df)

    features = []
    for col in ALL_FEATURES:
        kind = (
            "numeric" if col in NUMERIC_FEATURES
            else "categorical" if col in CATEGORICAL_FEATURES
            else "binary"
        )
        features.append({
            "name": col,
            "type": kind,
            "missing_pct": round(float(df[col].isna().mean()) * 100, 1),
        })

    return {
        "dataset": "Weather AUS - predict RainTomorrow",
        "target": TARGET,
        "n_rows": int(len(df)),
        "positive_rate_pct": round(float((df[TARGET] == "Yes").mean()) * 100, 1),
        "split": {
            "train": {"rows": int(len(train)), "years": _years_label(train)},
            "validation": {"rows": int(len(val)), "years": _years_label(val)},
            "test": {"rows": int(len(test)), "years": _years_label(test), "hidden": True},
        },
        "features": features,
    }
