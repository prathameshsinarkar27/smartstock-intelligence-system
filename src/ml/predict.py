"""
predict.py

Generates ensembled trend predictions and risk scores for companies
and upserts the results into the predictions table.

Usage:
    python -m src.ml.predict
    python -m src.ml.predict --symbols AAPL MSFT
"""

import argparse

import numpy as np
from psycopg2.extras import execute_values

from src.ml.feature_engineering import (
    FEATURE_COLUMNS,
    INT_TO_LABEL,
    LABEL_TO_INT,
    build_feature_dataset,
    build_latest_inference_rows,
)
from src.ml.train_model import ModelNotTrainedError, load_trained_models
from src.utils.database import get_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)

_UPSERT_QUERY = """
    INSERT INTO predictions (company_id, prediction_date, trend_prediction, risk_score)
    VALUES %s
    ON CONFLICT (company_id, prediction_date) DO UPDATE SET
        trend_prediction = EXCLUDED.trend_prediction,
        risk_score = EXCLUDED.risk_score,
        created_at = now();
"""


def generate_predictions(symbols: list[str] | None = None) -> list[tuple]:
    """
    Generate ensembled trend and risk predictions for each company.

    Args:
        symbols: Optional ticker symbols to restrict predictions to.

    Returns:
        Tuples containing company ID, prediction date, trend, and risk score.

    Raises:
        ModelNotTrainedError: If trained models are unavailable.
    """
    rf_model, xgb_model, _metadata = load_trained_models()

    dataset = build_feature_dataset(symbols)
    latest_rows = build_latest_inference_rows(dataset)

    if latest_rows.empty:
        logger.warning("No companies have a complete latest feature row to predict from.")
        return []

    X = latest_rows[FEATURE_COLUMNS]
    rf_proba = rf_model.predict_proba(X)
    xgb_proba = xgb_model.predict_proba(X)
    ensemble_proba = (rf_proba + xgb_proba) / 2.0

    predicted_class_indices = np.argmax(ensemble_proba, axis=1)
    down_class_index = LABEL_TO_INT["down"]

    results = []
    for row_position, (_, row) in enumerate(latest_rows.iterrows()):
        trend_prediction = INT_TO_LABEL[predicted_class_indices[row_position]]
        risk_score = float(ensemble_proba[row_position, down_class_index])
        results.append((int(row["company_id"]), row["date"], trend_prediction, risk_score))

    return results


def write_predictions(predictions: list[tuple]) -> int:
    """
    Upsert generated predictions into the predictions table.

    Args:
        predictions: Output of generate_predictions().

    Returns:
        Number of rows upserted.
    """
    if not predictions:
        logger.warning("write_predictions: nothing to write.")
        return 0

    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, _UPSERT_QUERY, predictions)

    logger.info("Upserted %d prediction row(s).", len(predictions))
    return len(predictions)


def run_predictions(symbols: list[str] | None = None) -> int:
    """
    Generate and store predictions for matching companies.

    Args:
        symbols: Optional ticker symbols to restrict predictions to.

    Returns:
        Number of predictions upserted.

    Raises:
        ModelNotTrainedError: If trained models are unavailable.
    """
    predictions = generate_predictions(symbols)
    return write_predictions(predictions)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate ensembled trend/risk predictions and write them to the predictions table."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Restrict predictions to these ticker symbols, e.g. --symbols AAPL MSFT. "
        "Default: every company with enough loaded history.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the prediction pipeline."""
    args = parse_args()
    try:
        written = run_predictions(symbols=args.symbols)
        logger.info("Prediction run complete. %d row(s) written.", written)
    except ModelNotTrainedError as exc:
        logger.error("Prediction aborted: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
