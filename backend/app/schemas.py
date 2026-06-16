"""
The training-request "order form".

When the frontend asks the backend to train a model, it sends a JSON body. This
file defines EXACTLY what that body is allowed to contain. Every knob from the
README's config table is one field below.

Pydantic (via `BaseModel`) reads the type annotations here and, for every incoming
request, automatically:
  - checks each field is the right type and an allowed value,
  - fills in defaults for anything the user left out,
  - rejects bad input with a clear error (FastAPI turns this into a 422 response).

So this single class is our validation layer AND our documentation — FastAPI uses
it to build the interactive /docs page too.
"""
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .data import ALL_FEATURES


class TrainConfig(BaseModel):
    """All user-configurable knobs for one training run.

    Each field maps 1:1 to a row in the README's "User-configurable knobs" table.
    The value after `=` (or inside `Field(...)`) is the DEFAULT used when the
    request omits that field — these defaults reproduce the notebook's baseline.
    """

    # --- Feature selection -------------------------------------------------
    # Which of the 21 dataset columns to feed the model. Default: all of them.
    # `default_factory=lambda: list(ALL_FEATURES)` makes a fresh copy of the list
    # for each new config (you must never share one mutable list as a default).
    features: list[str] = Field(default_factory=lambda: list(ALL_FEATURES))

    # --- Missing values ----------------------------------------------------
    # What to do with cells that have no value.
    #   drop_row  = delete any row that has a missing value
    #   drop_col  = delete any column that has missing values
    #   impute    = fill the gap with a statistic (the default, and most useful)
    missing_strategy: Literal["drop_row", "drop_col", "impute"] = "impute"

    # If imputing, which statistic fills the gap. Only used when
    # missing_strategy == "impute" (ignored otherwise).
    #   mean/median = for numeric columns;  mode = most common value;
    #   constant    = a fixed fill value (e.g. 0)
    impute_statistic: Literal["mean", "median", "mode", "constant"] = "median"

    # Columns that should ALSO get a "<col>_was_missing" 0/1 flag added, so the
    # model can learn from the *fact* that a value was missing. A list of column
    # names; empty list = no flags (the default).
    missing_indicator_columns: list[str] = Field(default_factory=list)

    # --- Encoding (turning text categories into numbers) -------------------
    # How to encode the `Location` column (49 weather stations).
    #   onehot  = one 0/1 column per location (default)
    #   ordinal = a single integer code per location
    #   drop    = don't use Location at all
    location_encoding: Literal["onehot", "ordinal", "drop"] = "onehot"

    # How to encode the 3 wind-direction columns (N, NNE, ... a compass).
    #   cyclical = sin/cos pair so adjacent directions stay close (default; what
    #              the notebook used)
    #   onehot   = one 0/1 column per direction
    #   drop     = don't use wind direction
    wind_encoding: Literal["onehot", "cyclical", "drop"] = "cyclical"

    # --- Scaling -----------------------------------------------------------
    # Put continuous features on a comparable scale (important for logistic
    # regression so coefficients and regularization are fair).
    #   none / standard (mean0,sd1) / minmax (0..1) / robust (median/IQR)
    scaling: Literal["none", "standard", "minmax", "robust"] = "standard"

    # --- Imbalance ---------------------------------------------------------
    # Only ~22% of days are "rain tomorrow". `balanced` re-weights the rare
    # class so the model doesn't just predict "no rain" every time.
    class_weight: Literal["none", "balanced"] = "balanced"

    # --- Leakage policy ----------------------------------------------------
    # The teaching knob. Decides which data the imputer/scaler/encoder LEARN from:
    #   prevent = fit on train only (correct)
    #   allow   = fit on train + validation (the classic mistake — inflates the
    #             validation score; the hidden test set later exposes it)
    leakage_policy: Literal["prevent", "allow"] = "prevent"

    # --- Logistic Regression hyperparameters -------------------------------
    # Regularization strength. Smaller C = stronger regularization (simpler model).
    # Must be > 0. The UI shows this on a log-scale slider.
    C: float = Field(default=1.0, gt=0.0)

    # Type of regularization penalty. L2 (ridge) is the default; L1 (lasso) can
    # zero out weak features.
    penalty: Literal["L1", "L2"] = "L2"

    # Probability cutoff for calling a prediction "rain". 0.5 is the default;
    # lowering it catches more rainy days (higher recall) at the cost of more
    # false alarms. Must be between 0 and 1.
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    # Reject any field name we didn't define above, instead of silently ignoring
    # it. Catches frontend typos early (e.g. "Cee" instead of "C").
    model_config = {"extra": "forbid"}

    # --- Custom validation -------------------------------------------------
    # `Literal` and number ranges are checked automatically, but "is every name
    # in `features` a real dataset column?" needs a small custom check. A
    # field_validator runs after the basic type check and can reject bad values.
    @field_validator("features")
    @classmethod
    def features_must_be_known(cls, value: list[str]) -> list[str]:
        unknown = [f for f in value if f not in ALL_FEATURES]
        if unknown:
            raise ValueError(f"Unknown feature(s): {unknown}. Allowed: {ALL_FEATURES}")
        if not value:
            raise ValueError("Select at least one feature.")
        return value
