"""
Run one training experiment and compute the metric panel.

This is the "cooking" step. `build_pipeline` (in pipeline.py) gave us an unfitted
recipe; here we:
  1. load the data and make the fixed chronological split,
  2. apply the missing-value strategy that removes data (drop_row / drop_col),
  3. fit the recipe, HONOURING the leakage knob (which data the cleaning learns from),
  4. train the model and score it at the user's threshold,
  5. return the full metric panel for train + validation.

The hidden TEST split is deliberately untouched here — it's only revealed by a
separate "final evaluation" call later, so the validation score stays the honest
thing the user tunes against.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .data import TARGET, load_raw, split_chronological
from .pipeline import build_pipeline
from .schemas import TrainConfig


def _target(part: pd.DataFrame) -> pd.Series:
    """Turn the Yes/No RainTomorrow column into 1/0 labels."""
    return (part[TARGET] == "Yes").astype(int)


def _metrics(y_true: pd.Series, proba: np.ndarray, threshold: float) -> dict:
    """The full scorecard for one split.

    `proba` is the model's predicted P(rain) for each row. Label-based metrics
    (accuracy/precision/recall/F1/confusion) depend on the threshold; the AUC
    metrics summarise the probabilities across ALL thresholds, so they don't.
    """
    y_pred = (proba >= threshold).astype(int)
    # labels=[0, 1] guarantees a full 2x2 even if the model predicts only one class.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "positive_rate": round(float(y_true.mean()), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        # zero_division=0: if the model predicts no positives, call precision 0 (not an error).
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def run_training(config: TrainConfig) -> dict:
    """Train one model from a validated config and return its metric panel."""
    df = load_raw()
    train, val, _test = split_chronological(df)   # test stays hidden

    features = list(config.features)

    # --- Missing-value strategies that REMOVE data (handled before the pipeline) ---
    # The pipeline's imputers only run for missing_strategy == "impute"; the two
    # "drop" modes change the data/columns themselves, so we do them here.
    if config.missing_strategy == "drop_col":
        # Drop any selected feature that has a blank anywhere in the training split.
        features = [f for f in features if not train[f].isna().any()]
        if not features:
            raise ValueError("drop_col removed every selected feature (all had missing values).")
        config = config.model_copy(update={"features": features})

    if config.missing_strategy == "drop_row":
        # Drop rows with a blank in any selected feature (independently per split).
        train = train.dropna(subset=features)
        val = val.dropna(subset=features)

    y_train, y_val = _target(train), _target(val)

    # --- Build the recipe, then split it into its two halves so we can control
    #     exactly what the preprocessing learns from (the leakage knob). ---
    pipe = build_pipeline(config)
    preprocess = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]

    # LEAKAGE KNOB: which rows do the imputer/scaler/encoder learn their stats from?
    #   prevent -> train only (correct)
    #   allow   -> train + validation (the mistake: validation peeks into the cleaning)
    # Either way the MODEL itself is only ever trained on the training rows.
    if config.leakage_policy == "allow":
        preprocess.fit(pd.concat([train, val]))
    else:
        preprocess.fit(train)

    X_train = preprocess.transform(train)
    X_val = preprocess.transform(val)

    model.fit(X_train, y_train)

    # Predicted probability of "rain tomorrow" for each row (column 1 = positive class).
    p_train = model.predict_proba(X_train)[:, 1]
    p_val = model.predict_proba(X_val)[:, 1]

    return {
        "config": config.model_dump(),
        "n_features_in": X_train.shape[1],   # width after encoding/scaling
        "train": _metrics(y_train, p_train, config.threshold),
        "validation": _metrics(y_val, p_val, config.threshold),
    }
