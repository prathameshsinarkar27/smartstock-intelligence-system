"""
train_model.py

Trains two classifiers — a scikit-learn RandomForestClassifier and an
XGBoost XGBClassifier — on the pooled, cross-company feature dataset from
src/ml/feature_engineering.py, to predict the forward trend label
("up"/"down"/"flat", see feature_engineering.py's module docstring for how
that label is defined).

Both models are trained and saved (rather than picking a single "best"
one) because predict.py uses them together: it averages their predicted
class probabilities (a simple, standard ensembling approach) rather than
committing to whichever one scored marginally higher on a single holdout
split. evaluate_model.py reports metrics for each model individually and
for that same ensemble, so the ensembling choice is itself verifiable.

The train/test split is chronological, not random: every row at or before
a cutoff date is training data, everything after is the holdout test set.
A random split would leak information (a model can partly memorize
company-specific price regimes it saw in "test" rows from the same week
as "train" rows for the same company), which a random train_test_split
would not catch but a realistic backtest-style split does.

Usage (from project root, with venv activated):
    python -m src.ml.train_model
    python -m src.ml.train_model --symbols AAPL MSFT JPM
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from src.ml.feature_engineering import (
    FEATURE_COLUMNS,
    LABEL_TO_INT,
    build_feature_dataset,
    build_training_rows,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
RANDOM_FOREST_PATH = MODELS_DIR / "random_forest_trend_model.joblib"
XGBOOST_PATH = MODELS_DIR / "xgboost_trend_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Fraction of (chronologically-sorted) rows held out as the test set.
TEST_SIZE = 0.2

RANDOM_STATE = 42


class TrainingDataError(Exception):
    """Raised when there isn't enough usable data to train a model."""


class ModelNotTrainedError(Exception):
    """Raised when evaluate_model.py or predict.py is run before train_model.py."""


def load_trained_models() -> tuple:
    """
    Load the saved RandomForest model, XGBoost model, and training
    metadata from models/. Shared by evaluate_model.py and predict.py so
    there's exactly one place that knows the on-disk artifact layout.

    Returns:
        A (rf_model, xgb_model, metadata) tuple.

    Raises:
        ModelNotTrainedError: If any of the three expected files is
            missing — i.e. train_model.py hasn't been run yet.
    """
    missing = [
        str(path) for path in (RANDOM_FOREST_PATH, XGBOOST_PATH, METADATA_PATH) if not path.exists()
    ]
    if missing:
        raise ModelNotTrainedError(
            f"Missing trained model file(s): {', '.join(missing)}. "
            f"Run `python -m src.ml.train_model` first."
        )

    rf_model = joblib.load(RANDOM_FOREST_PATH)
    xgb_model = joblib.load(XGBOOST_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return rf_model, xgb_model, metadata


def chronological_split(train_rows, test_size: float = TEST_SIZE):
    """
    Split a training dataset into train/test sets by date, not randomly.

    Args:
        train_rows: Output of build_training_rows() — every row has a
            complete feature set and a non-null label.
        test_size: Fraction of rows to hold out as the test set, taken
            from the most recent dates.

    Returns:
        A (train_df, test_df, cutoff_date) tuple. Every row in train_df
        has date <= cutoff_date; every row in test_df has date >
        cutoff_date. cutoff_date is returned so it can be recorded in
        model_metadata.json — evaluate_model.py uses it to reconstruct
        the same held-out test set later.

    Raises:
        TrainingDataError: If there are too few distinct dates to form a
            non-empty train and test split (fewer than 5 unique dates).
    """
    unique_dates = sorted(train_rows["date"].unique())

    if len(unique_dates) < 5:
        raise TrainingDataError(
            f"Only {len(unique_dates)} distinct date(s) with usable training rows — "
            f"need more price/news history loaded before training. Run the ingestion "
            f"and ETL pipeline for more historical data, then try again."
        )

    cutoff_index = max(1, int(len(unique_dates) * (1 - test_size)) - 1)
    cutoff_date = unique_dates[cutoff_index]

    train_df = train_rows[train_rows["date"] <= cutoff_date].reset_index(drop=True)
    test_df = train_rows[train_rows["date"] > cutoff_date].reset_index(drop=True)

    if train_df.empty or test_df.empty:
        raise TrainingDataError(
            "Chronological split produced an empty train or test set — need more "
            "historical data spread across more dates before training."
        )

    return train_df, test_df, cutoff_date


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """
    Fit a RandomForestClassifier with settings suited to a small tabular
    financial feature set: enough trees to stabilize predictions, a
    depth cap to reduce overfitting on a modest number of training rows,
    and class_weight="balanced" since "flat" rows typically outnumber
    "up"/"down" rows given the threshold-based label definition.

    Args:
        X_train: Feature matrix (FEATURE_COLUMNS order), training rows only.
        y_train: Integer-encoded labels (see LABEL_TO_INT), training rows only.

    Returns:
        The fitted classifier.
    """
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train) -> XGBClassifier:
    """
    Fit an XGBClassifier for the same 3-class trend prediction task.

    Args:
        X_train: Feature matrix (FEATURE_COLUMNS order), training rows only.
        y_train: Integer-encoded labels (see LABEL_TO_INT), training rows only.

    Returns:
        The fitted classifier.
    """
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=len(LABEL_TO_INT),
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def run_training(symbols: list[str] | None = None) -> dict:
    """
    End-to-end training run: build the feature dataset, split it
    chronologically, train both classifiers, save them plus metadata to
    models/, and return that metadata.

    Args:
        symbols: If provided, restrict training data to these ticker
            symbols. If None, every company with enough history is used.

    Returns:
        The metadata dict that was written to model_metadata.json (see
        that file for its exact shape) — feature_columns, label_to_int,
        test_cutoff_date, trained_at, row counts, and each model's
        holdout accuracy.

    Raises:
        TrainingDataError: If build_feature_dataset() yields no usable
            training rows, or chronological_split() can't form a valid
            train/test split (see chronological_split's docstring).
    """
    dataset = build_feature_dataset(symbols)
    train_rows = build_training_rows(dataset)

    if train_rows.empty:
        raise TrainingDataError(
            "No usable training rows found. Make sure price and (ideally) news/"
            "sentiment data has been loaded for your tracked symbols before "
            "running this script."
        )

    logger.info(
        "Built %d usable training row(s) across %d companies.",
        len(train_rows),
        train_rows["symbol"].nunique(),
    )

    train_df, test_df, cutoff_date = chronological_split(train_rows)
    logger.info(
        "Chronological split: %d train row(s) through %s, %d test row(s) after.",
        len(train_df), cutoff_date, len(test_df),
    )

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["label"].map(LABEL_TO_INT)
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["label"].map(LABEL_TO_INT)

    logger.info("Training RandomForestClassifier...")
    rf_model = train_random_forest(X_train, y_train)
    rf_holdout_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
    logger.info("RandomForest holdout accuracy: %.4f", rf_holdout_accuracy)

    logger.info("Training XGBClassifier...")
    xgb_model = train_xgboost(X_train, y_train)
    xgb_holdout_accuracy = accuracy_score(y_test, xgb_model.predict(X_test))
    logger.info("XGBoost holdout accuracy: %.4f", xgb_holdout_accuracy)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_model, RANDOM_FOREST_PATH)
    joblib.dump(xgb_model, XGBOOST_PATH)
    logger.info("Saved models to %s and %s", RANDOM_FOREST_PATH, XGBOOST_PATH)

    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "label_to_int": LABEL_TO_INT,
        "test_cutoff_date": str(cutoff_date),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_companies": sorted(train_df["symbol"].unique().tolist()),
        "random_forest_holdout_accuracy": rf_holdout_accuracy,
        "xgboost_holdout_accuracy": xgb_holdout_accuracy,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved training metadata to %s", METADATA_PATH)

    return metadata


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for standalone script execution."""
    parser = argparse.ArgumentParser(
        description="Train Random Forest and XGBoost trend classifiers on the pooled feature dataset."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Restrict training data to these ticker symbols, e.g. --symbols AAPL MSFT. "
        "Default: every company with enough loaded history.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for standalone script execution."""
    args = parse_args()
    try:
        run_training(symbols=args.symbols)
    except TrainingDataError as exc:
        logger.error("Training aborted: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
