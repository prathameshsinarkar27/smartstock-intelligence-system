"""
shap_analysis.py

Explains a single company's ML prediction in
terms of which features pushed it toward or away from the "down" class —
the same class risk_score is defined over — using SHAP
(SHapley Additive exPlanations) TreeExplainers over the saved RandomForest
and XGBoost models.

A unit-consistency subtlety drives this module's design, and is worth
understanding before reading the code:

    shap.TreeExplainer's default output is in each model's own natural
    space. For this project's RandomForestClassifier, that happens to be
    probability space (its expected_value across classes sums to ~1). For
    this project's XGBClassifier, TreeExplainer's default output is raw
    margin (pre-softmax log-odds) space — its expected_value does not sum
    to 1. Requesting probability-space output directly from XGBoost's
    TreeExplainer (model_output="probability") isn't reliably supported
    for this project's shap/xgboost version combination.

    Averaging those two vectors directly, the way predict.py averages
    predict_proba() outputs, would silently mix incompatible units (a
    probability-space number added to a log-odds-space number) and
    produce a meaningless total. Instead, each model's SHAP vector for a
    given prediction is normalized to that model's own *share of its
    total absolute attribution* (i.e. "what fraction of this model's
    reasoning came from this feature, and in which direction") before the
    two models' shares are averaged. This keeps the combination
    dimensionless and honest about what it represents: relative feature
    influence within the ensemble's reasoning, not a probability
    decomposition.

Models are loaded and their SHAP explainers built once per process
(_get_cached_explainers()) rather than per call — TreeExplainer
construction walks every tree, which isn't free to repeat on every page
load.
"""

from typing import Any

import numpy as np
import pandas as pd
import shap

from src.ml.feature_engineering import (
    FEATURE_COLUMNS,
    LABEL_TO_INT,
    build_feature_dataset,
    build_latest_inference_rows,
)
from src.ml.train_model import ModelNotTrainedError, load_trained_models
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TARGET_CLASS = "down"
DEFAULT_TOP_N = 5

# Friendlier labels for the dashboard than the raw column names — purely
# presentational, has no effect on the underlying SHAP computation.
FEATURE_DISPLAY_NAMES = {
    "close_to_sma_20": "Price vs 20-day average",
    "close_to_sma_50": "Price vs 50-day average",
    "close_to_ema_12": "Price vs 12-day EMA",
    "close_to_ema_26": "Price vs 26-day EMA",
    "rsi_14": "RSI (14-day)",
    "macd_histogram_norm": "MACD histogram",
    "bollinger_percent_b": "Bollinger Band position",
    "bollinger_bandwidth": "Bollinger Band width",
    "return_1d": "1-day return",
    "return_5d": "5-day return",
    "volume_ratio_20d": "Volume vs 20-day average",
    "sentiment_7d": "7-day news sentiment",
}

# Cache populated lazily by _get_cached_explainers(); avoids rebuilding
# TreeExplainers (which walk every tree in both models) on every call.
_cache: dict[str, Any] = {}


def _get_cached_explainers() -> tuple:
    """
    Load the trained models (if not already cached) and build a
    shap.TreeExplainer for each.

    Returns:
        A (rf_model, xgb_model, rf_explainer, xgb_explainer, metadata) tuple.

    Raises:
        ModelNotTrainedError: If train_model.py hasn't been run yet.
    """
    if not _cache:
        rf_model, xgb_model, metadata = load_trained_models()
        _cache["rf_model"] = rf_model
        _cache["xgb_model"] = xgb_model
        _cache["metadata"] = metadata
        _cache["rf_explainer"] = shap.TreeExplainer(rf_model)
        _cache["xgb_explainer"] = shap.TreeExplainer(xgb_model)
        logger.info("Built SHAP TreeExplainers for RandomForest and XGBoost.")

    return (
        _cache["rf_model"],
        _cache["xgb_model"],
        _cache["rf_explainer"],
        _cache["xgb_explainer"],
        _cache["metadata"],
    )


def clear_cache() -> None:
    """
    Drop the cached models/explainers, forcing the next call to
    _get_cached_explainers() to reload from disk. Used after retraining
    (train_model.py has produced new .joblib files) and in tests.
    """
    _cache.clear()


def _shap_values_for_class(explainer: shap.TreeExplainer, X: pd.DataFrame, class_index: int) -> np.ndarray:
    """
    Compute SHAP values for a single row and slice out one class.

    Args:
        explainer: A fitted shap.TreeExplainer.
        X: A single-row DataFrame in FEATURE_COLUMNS order.
        class_index: Which class's SHAP values to extract (see
            src.ml.feature_engineering.LABEL_TO_INT).

    Returns:
        A 1-D array of length len(FEATURE_COLUMNS): this model's raw SHAP
        value for each feature, for the requested class, in that model's
        own natural output space (see module docstring).
    """
    raw = np.asarray(explainer.shap_values(X))
    # Expected shape (1, n_features, n_classes) for a single-row multiclass
    # TreeExplainer call, matching every case exercised in this project's
    # tests. Squeeze the row dimension and select the class.
    return raw[0, :, class_index]


def _normalize_to_shares(shap_values: np.ndarray) -> np.ndarray:
    """
    Rescale a model's raw SHAP vector to each feature's share of that
    model's total absolute attribution for this prediction (see module
    docstring for why this, rather than raw units, is what gets averaged
    across the ensemble).

    Args:
        shap_values: Raw SHAP values for one model, one class, one row.

    Returns:
        An array the same shape as `shap_values`, rescaled so the sum of
        absolute values is 1.0. All-zero (every feature contributed
        nothing — only possible in degenerate cases) returns the input
        unchanged, i.e. all zeros.
    """
    total_absolute = np.abs(shap_values).sum()
    if total_absolute == 0:
        return shap_values
    return shap_values / total_absolute


def explain_row(feature_values: dict[str, float], target_class: str = DEFAULT_TARGET_CLASS,
                top_n: int = DEFAULT_TOP_N) -> dict[str, Any]:
    """
    Explain a single feature row's ensemble prediction for one class.

    Args:
        feature_values: A dict with every key in FEATURE_COLUMNS mapped to
            its numeric value for this row (e.g. one row of
            src.ml.feature_engineering.build_latest_inference_rows()'s
            output).
        target_class: Which class to explain — "up", "down", or "flat".
            Defaults to "down", matching risk_score's definition.
        top_n: How many top-contributing features to return.

    Returns:
        A dict with:
            - target_class: echoed back for convenience.
            - contributions: a list of up to top_n dicts, sorted by
              descending absolute contribution_share, each with
              feature (raw column name), display_name
              (FEATURE_DISPLAY_NAMES, falling back to the raw name),
              value (this row's raw feature value), contribution_share
              (this feature's averaged, normalized share of the
              ensemble's attribution toward target_class — positive
              pushes toward target_class, negative pushes away; see
              module docstring for what "share" means here), abs_share
              (its absolute value, for sizing a bar in the template
              without a Jinja abs filter), and direction ("positive" or
              "negative", for coloring that bar).

    Raises:
        ModelNotTrainedError: If train_model.py hasn't been run yet.
        KeyError: If `feature_values` is missing any FEATURE_COLUMNS key.
    """
    rf_model, xgb_model, rf_explainer, xgb_explainer, _metadata = _get_cached_explainers()
    class_index = LABEL_TO_INT[target_class]

    X = pd.DataFrame([{col: feature_values[col] for col in FEATURE_COLUMNS}])[FEATURE_COLUMNS]

    rf_shap = _shap_values_for_class(rf_explainer, X, class_index)
    xgb_shap = _shap_values_for_class(xgb_explainer, X, class_index)

    rf_shares = _normalize_to_shares(rf_shap)
    xgb_shares = _normalize_to_shares(xgb_shap)
    ensemble_shares = (rf_shares + xgb_shares) / 2.0

    ranked_indices = np.argsort(-np.abs(ensemble_shares))[:top_n]

    contributions = [
        {
            "feature": FEATURE_COLUMNS[i],
            "display_name": FEATURE_DISPLAY_NAMES.get(FEATURE_COLUMNS[i], FEATURE_COLUMNS[i]),
            "value": float(X.iloc[0, i]),
            "contribution_share": float(ensemble_shares[i]),
            "abs_share": float(abs(ensemble_shares[i])),
            "direction": "positive" if ensemble_shares[i] > 0 else "negative",
        }
        for i in ranked_indices
    ]

    return {"target_class": target_class, "contributions": contributions}


def explain_company_prediction(symbol: str, target_class: str = DEFAULT_TARGET_CLASS,
                                top_n: int = DEFAULT_TOP_N) -> dict[str, Any] | None:
    """
    Explain a single company's most recent prediction — the same feature
    row src/ml/predict.py would use for inference right now.

    Args:
        symbol: Stock ticker symbol, e.g. "AAPL".
        target_class: Which class to explain. Defaults to "down".
        top_n: How many top-contributing features to return.

    Returns:
        None if the company doesn't have a usable latest feature row
        (same condition under which predict.py would skip it — see
        src.ml.feature_engineering.build_latest_inference_rows). Otherwise
        explain_row()'s output, with "symbol" and "prediction_date"
        (the feature row's date) added.

    Raises:
        ModelNotTrainedError: If train_model.py hasn't been run yet.
    """
    dataset = build_feature_dataset([symbol])
    latest_rows = build_latest_inference_rows(dataset)

    if latest_rows.empty:
        return None

    row = latest_rows.iloc[0]
    feature_values = {col: row[col] for col in FEATURE_COLUMNS}

    explanation = explain_row(feature_values, target_class=target_class, top_n=top_n)
    explanation["symbol"] = symbol
    explanation["prediction_date"] = row["date"]
    return explanation
