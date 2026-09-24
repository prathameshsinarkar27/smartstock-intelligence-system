"""
train_model.py

Trains RandomForest and XGBoost classifiers on the pooled feature dataset
to predict forward trend labels ("up", "down", "flat").

Models are saved separately and used together by predict.py, which averages
their class probabilities. Evaluation reports metrics for both models and
the ensemble.

The train/test split is chronological to prevent future-data leakage.
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

# Fraction of chronologically sorted rows held out for testing.
TEST_SIZE = 0.2

RANDOM_STATE = 42


class TrainingDataError(Exception):
    """Raised when there is insufficient usable data for training."""


class ModelNotTrainedError(Exception):
    """Raised when trained model artifacts are unavailable."""


def load_trained_models() -> tuple:
    """
    Load the saved RandomForest, XGBoost, and training metadata.

    Returns:
        A (rf_model, xgb_model, metadata) tuple.

    Raises:
        ModelNotTrainedError: If any expected model artifact is missing.
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
    Split training data chronologically.

    Args:
        train_rows: Training rows with complete features and labels.
        test_size: Fraction of recent dates reserved for testing.

    Returns:
        A (train_df, test_df, cutoff_date) tuple.

    Raises:
        TrainingDataError: If there are too few dates or an empty split.
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
    Fit a RandomForestClassifier for the 3-class trend task.

    Args:
        X_train: Training feature matrix.
        y_train: Integer-encoded training labels.

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
    Fit an XGBClassifier for the 3-class trend task.

    Args:
        X_train: Training feature matrix.
        y_train: Integer-encoded training labels.

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
    Build features, train both classifiers, and save models and metadata.

    Args:
        symbols: Optional ticker symbols to restrict training data.

    Returns:
        Metadata written to model_metadata.json.

    Raises:
        TrainingDataError: If usable training data or a valid split is unavailable.
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
    """Parse command-line arguments."""
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
    """Run the training pipeline."""
    args = parse_args()
    try:
        run_training(symbols=args.symbols)
    except TrainingDataError as exc:
        logger.error("Training aborted: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
