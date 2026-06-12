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

# Lookup tables: a config string -> the scikit-learn object/argument it means.
_SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}
_IMPUTE_STRATEGY = {"mean": "mean", "median": "median", "mode": "most_frequent", "constant": "constant"}


# --- small helpers -------------------------------------------------------

def _make_imputer(config: TrainConfig) -> SimpleImputer:
    """A SimpleImputer fills gaps. For numeric cols we honour the user's choice;
    constant fills with 0."""
    strategy = _IMPUTE_STRATEGY[config.impute_statistic]
    if strategy == "constant":
        return SimpleImputer(strategy="constant", fill_value=0)
    return SimpleImputer(strategy=strategy)


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

def _numeric_recipe(config: TrainConfig):
    """Continuous columns: (optionally) impute, then (optionally) scale."""
    steps = []
    if config.missing_strategy == "impute":
        steps.append(("impute", _make_imputer(config)))
    scaler = _make_scaler(config.scaling)
    if scaler is not None:
        steps.append(("scale", scaler))
    return Pipeline(steps) if steps else "passthrough"


def _location_recipe(config: TrainConfig):
    """Location: fill blanks with the most common station, then encode to numbers."""
    if config.location_encoding == "onehot":
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:  # "ordinal" — one integer code per location
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    steps = []
    if config.missing_strategy == "impute":
        steps.append(("impute", SimpleImputer(strategy="most_frequent")))
    steps.append(("encode", encoder))
    return Pipeline(steps)


def _wind_recipe(config: TrainConfig):
    """Wind directions: either cyclical sin/cos (handles its own blanks) or one-hot."""
    if config.wind_encoding == "cyclical":
        return FunctionTransformer(_cyclical_wind)
    steps = []
    if config.missing_strategy == "impute":
        steps.append(("impute", SimpleImputer(strategy="most_frequent")))
    steps.append(("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
    return Pipeline(steps)


def _binary_recipe(config: TrainConfig):
    """RainToday is Yes/No text -> impute, then map to 1/0 (No=0, Yes=1)."""
    steps = []
    if config.missing_strategy == "impute":
        steps.append(("impute", SimpleImputer(strategy="most_frequent")))
    steps.append(("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))
    return Pipeline(steps)


# --- the public builder --------------------------------------------------

def build_pipeline(config: TrainConfig) -> Pipeline:
    """Assemble the full unfitted pipeline (preprocess + LogisticRegression)."""

    # 1) Sort the SELECTED features into groups, each routed to its own sub-recipe.
    #    A column the user didn't select simply never appears here, so it's ignored.
    transformers = []

    numeric = [f for f in config.features if f in NUMERIC_FEATURES]
    if numeric:
        transformers.append(("numeric", _numeric_recipe(config), numeric))

    if "Location" in config.features and config.location_encoding != "drop":
        transformers.append(("location", _location_recipe(config), ["Location"]))

    wind = [w for w in WIND_DIR_COLS if w in config.features]
    if wind and config.wind_encoding != "drop":
        transformers.append(("wind", _wind_recipe(config), wind))

    if "RainToday" in config.features:
        transformers.append(("binary", _binary_recipe(config), ["RainToday"]))

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
