"""
Turn a validated TrainConfig into a scikit-learn pipeline.

A scikit-learn "pipeline" is a recipe with two parts chained together:
  1. PREPROCESS — clean + transform the raw columns into a pure-number matrix
     (impute missing values, scale numbers, encode text categories).
  2. MODEL — the LogisticRegression that learns from that matrix.

The key idea: we build this recipe but DON'T run it here. `build_pipeline` returns
an *unfitted* pipeline. Sub-step 2c will call `.fit(...)` on the training data and
`.predict(...)` to score it. Keeping "build" and "fit" separate is what lets the
leakage knob later choose *which* data the recipe learns from.

Why a ColumnTransformer? Different columns need different treatment — you can't
scale the word "Albury" or one-hot-encode a temperature. ColumnTransformer routes
each group of columns to the right sub-recipe and glues the results side by side.
"""
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from .data import NUMERIC_FEATURES
from .schemas import TrainConfig

# The 3 wind-direction columns get special encoding; Location is handled on its own.
WIND_DIR_COLS = ["WindGustDir", "WindDir9am", "WindDir3pm"]

# 16-point compass -> bearing in degrees. Used by the cyclical wind encoder.
COMPASS = {
    "N": 0.0,   "NNE": 22.5,  "NE": 45.0,   "ENE": 67.5,
    "E": 90.0,  "ESE": 112.5, "SE": 135.0,  "SSE": 157.5,
    "S": 180.0, "SSW": 202.5, "SW": 225.0,  "WSW": 247.5,
    "W": 270.0, "WNW": 292.5, "NW": 315.0,  "NNW": 337.5,
}

# Lookup table: scaling name -> the scikit-learn class.
_SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}


# --- small helpers -------------------------------------------------------

def resolve_missing(config: TrainConfig, features: list[str]) -> dict[str, tuple[str, str]]:
    """Work out (strategy, statistic) for EACH feature.

    A feature the user configured uses that setting; any other feature falls back
    to the default (impute by median). Text columns can't use mean/median, so we
    quietly downgrade those to mode. Shared by the pipeline and the training step.
    """
    resolved = {}
    for feature in features:
        entry = config.missing.get(feature)
        strategy = entry.strategy if entry else "impute"
        statistic = entry.statistic if entry else "median"
        if feature not in NUMERIC_FEATURES and statistic in ("mean", "median"):
            statistic = "mode"
        resolved[feature] = (strategy, statistic)
    return resolved


def _numeric_imputer(statistic: str) -> SimpleImputer:
    """Imputer for a numeric column, per the chosen statistic."""
    if statistic == "constant":
        return SimpleImputer(strategy="constant", fill_value=0)
    if statistic == "mode":
        return SimpleImputer(strategy="most_frequent")
    return SimpleImputer(strategy=statistic)  # "mean" or "median"


def _text_imputer(statistic: str) -> SimpleImputer:
    """Imputer for a text column: only mode or a constant placeholder make sense."""
    if statistic == "constant":
        return SimpleImputer(strategy="constant", fill_value="__missing__")
    return SimpleImputer(strategy="most_frequent")


def _make_scaler(name: str):
    """Return a fresh scaler object, or None when scaling is off."""
    if name == "none":
        return None
    return _SCALERS[name]()


def _cyclical_wind(frame):
    """Convert wind-direction strings to sin/cos pairs.

    A compass is circular: N (0 degrees) and NNW (337.5 degrees) are neighbours,
    but as plain numbers 0 and 337.5 look far apart. Mapping each bearing to
    (sin, cos) places them on a circle so adjacent directions stay close.
    This is a fixed formula (no fitting), so it's identical on every split.
    Missing/unknown directions become (0, 0) — the neutral "no direction".
    """
    frame = pd.DataFrame(frame)
    columns = []
    for col in frame.columns:
        radians = np.deg2rad(frame[col].map(COMPASS).astype("float64"))
        columns.append(np.sin(radians).fillna(0.0).to_numpy())
        columns.append(np.cos(radians).fillna(0.0).to_numpy())
    return np.column_stack(columns)


# --- per-group sub-recipes ----------------------------------------------

# Note: every recipe always includes an imputer. That's safe because rows are
# already row-dropped (for "drop_row" features) and columns excluded (for
# "drop_col") back in training.py BEFORE this runs — so a leftover imputer on a
# gap-free column is simply a harmless no-op. The recipe only needs the statistic.

def _numeric_recipe(config: TrainConfig, statistic: str):
    """Continuous columns: impute (per statistic), then (optionally) scale."""
    steps = [("impute", _numeric_imputer(statistic))]
    scaler = _make_scaler(config.scaling)
    if scaler is not None:
        steps.append(("scale", scaler))
    return Pipeline(steps)


def _location_recipe(config: TrainConfig, statistic: str):
    """Location: fill blanks, then encode to numbers."""
    if config.location_encoding == "onehot":
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:  # "ordinal" — one integer code per location
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    return Pipeline([("impute", _text_imputer(statistic)), ("encode", encoder)])


def _wind_recipe(config: TrainConfig):
    """Wind directions: either cyclical sin/cos (handles its own blanks) or one-hot."""
    if config.wind_encoding == "cyclical":
        return FunctionTransformer(_cyclical_wind)
    return Pipeline([
        ("impute", _text_imputer("mode")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])


def _binary_recipe(statistic: str):
    """RainToday is Yes/No text -> impute, then map to 1/0 (No=0, Yes=1)."""
    return Pipeline([
        ("impute", _text_imputer(statistic)),
        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])


# --- the public builder --------------------------------------------------

def build_pipeline(config: TrainConfig) -> Pipeline:
    """Assemble the full unfitted pipeline (preprocess + LogisticRegression)."""

    # 1) Sort the SELECTED features into groups, each routed to its own sub-recipe.
    #    A column the user didn't select simply never appears here, so it's ignored.
    transformers = []

    # The per-feature missing settings (with defaults filled in).
    resolved = resolve_missing(config, config.features)

    # Numeric columns can each have a different impute statistic, so we group them
    # by statistic — one transformer per distinct statistic (e.g. all "median"
    # columns together, all "mean" columns together).
    numeric = [f for f in config.features if f in NUMERIC_FEATURES]
    by_statistic = defaultdict(list)
    for feature in numeric:
        by_statistic[resolved[feature][1]].append(feature)
    for statistic, cols in by_statistic.items():
        transformers.append((f"numeric_{statistic}", _numeric_recipe(config, statistic), cols))

    if "Location" in config.features and config.location_encoding != "drop":
        transformers.append(("location", _location_recipe(config, resolved["Location"][1]), ["Location"]))

    wind = [w for w in WIND_DIR_COLS if w in config.features]
    if wind and config.wind_encoding != "drop":
        transformers.append(("wind", _wind_recipe(config), wind))

    if "RainToday" in config.features:
        transformers.append(("binary", _binary_recipe(resolved["RainToday"][1]), ["RainToday"]))

    # Optional "was this value missing?" 0/1 flag columns for the chosen columns.
    if config.missing_indicator_columns:
        transformers.append(
            ("missing_flags", MissingIndicator(features="all"), config.missing_indicator_columns)
        )

    # `remainder="drop"` = any column not handled above is discarded.
    preprocess = ColumnTransformer(transformers, remainder="drop")

    # 2) The model. Modern scikit-learn picks the penalty via `l1_ratio`:
    #    0 = pure L2 (ridge), 1 = pure L1 (lasso). L1 needs the 'liblinear'
    #    solver; L2 works with the default 'lbfgs'.
    is_l1 = config.penalty == "L1"
    model = LogisticRegression(
        C=config.C,
        l1_ratio=1 if is_l1 else 0,
        solver="liblinear" if is_l1 else "lbfgs",
        class_weight=None if config.class_weight == "none" else "balanced",
        max_iter=1000,                            # raised so it reliably converges
        random_state=42,                          # reproducible results
    )

    # 3) Chain them: raw columns -> preprocess -> model.
    return Pipeline([("preprocess", preprocess), ("model", model)])
