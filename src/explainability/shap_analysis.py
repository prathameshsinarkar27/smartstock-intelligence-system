"""
shap_analysis.py

Explains a company's ML prediction using SHAP feature attributions from
the saved RandomForest and XGBoost models.

The two models produce SHAP values in different output spaces, so their
raw values cannot be averaged directly. Each model's SHAP vector is
normalized to feature shares of total absolute attribution before the
shares are averaged. This produces dimensionless relative feature
influence rather than a probability decomposition.

SHAP explainers are cached and reused to avoid rebuilding tree explainers
on every call.
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

# Dashboard-friendly labels for feature columns.
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

# Lazily populated model and explainer cache.
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
    """Clear cached models and explainers."""
    _cache.clear()


def _shap_values_for_class(explainer: shap.TreeExplainer, X: pd.DataFrame, class_index: int) -> np.ndarray:
    """Compute SHAP values for one row and return values for one class."""
    raw = np.asarray(explainer.shap_values(X))
    # Expected shape (1, n_features, n_classes) for a single-row multiclass
    # TreeExplainer call, matching every case exercised in this project's
    # tests. Squeeze the row dimension and select the class.
    return raw[0, :, class_index]


def _normalize_to_shares(shap_values: np.ndarray) -> np.ndarray:
    """
    Normalize SHAP values to each feature's share of total absolute
    attribution.
    """
    total_absolute = np.abs(shap_values).sum()
    if total_absolute == 0:
        return shap_values
    return shap_values / total_absolute


def explain_row(feature_values: dict[str, float], target_class: str = DEFAULT_TARGET_CLASS,
                top_n: int = DEFAULT_TOP_N) -> dict[str, Any]:
    """
    Explain an ensemble prediction for one feature row.

    Returns the top contributing features with their values, normalized
    contribution shares, absolute shares, and directions.
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
    Explain the most recent prediction for a company.

    Returns None when no usable latest feature row is available.
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
