"""
predict.py

Generates a trend prediction and risk score for each company's most
recent complete feature row, and upserts the results into the
`predictions` table (database/tables.sql, Phase 2), for the dashboard's
ML Risk Score KPI card and ML Predictions section (Phase 8).

Prediction approach: both saved models (RandomForest, XGBoost — see
train_model.py) are loaded, each produces a predicted probability for
"up"/"down"/"flat", and the two probability vectors are averaged (simple
ensembling — see train_model.py's module docstring for why both models
are kept rather than picking one "winner"). The final trend_prediction is
whichever class has the highest averaged probability; risk_score is that
same averaged probability for the "down" class specifically — i.e. "how
likely does the ensemble think a downward move is over the next
FORWARD_HORIZON_DAYS trading days," which is what the predictions table's
CHECK constraint (0 <= risk_score <= 1) and the dashboard's "ML Risk
Score" label both expect.

prediction_date is set to the date of the feature row used (the most
recent date with a full trading history for that company), not "today" —
consistent with how src/analytics/kpi_calculator.py treats its own
period_end_date: it's the last date the underlying data actually
supports, not the wall-clock date the script happened to run on.

Usage (from project root, with venv activated):
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
    Build the latest feature row for each matching company and produce an
    ensembled trend/risk prediction for it.

    Args:
        symbols: If provided, restrict to these ticker symbols. If None,
            every company with enough history is used.

    Returns:
        A list of (company_id, prediction_date, trend_prediction,
        risk_score) tuples, one per company that has a usable latest
        feature row. Companies without enough price history yet (see
        feature_engineering.MIN_PRICE_HISTORY_ROWS) are silently absent
        — there's nothing to predict from.

    Raises:
        ModelNotTrainedError: If train_model.py hasn't been run yet.
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
        predictions: Output of generate_predictions() — (company_id,
            prediction_date, trend_prediction, risk_score) tuples.

    Returns:
        The number of rows upserted. 0 if `predictions` is empty (no
        database call is made in that case).
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
    End-to-end prediction run: generate ensembled predictions for every
    matching company and write them to the database.

    Args:
        symbols: If provided, restrict to these ticker symbols.

    Returns:
        The number of predictions.py rows upserted.

    Raises:
        ModelNotTrainedError: If train_model.py hasn't been run yet.
    """
    predictions = generate_predictions(symbols)
    return write_predictions(predictions)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for standalone script execution."""
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
    """Entry point for standalone script execution."""
    args = parse_args()
    try:
        written = run_predictions(symbols=args.symbols)
        logger.info("Prediction run complete. %d row(s) written.", written)
    except ModelNotTrainedError as exc:
        logger.error("Prediction aborted: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
