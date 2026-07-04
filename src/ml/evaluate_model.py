"""
evaluate_model.py

Loads the models and metadata saved by train_model.py, reconstructs the
exact same chronological held-out test set (using the test_cutoff_date
recorded in model_metadata.json), and reports classification metrics for
the RandomForest model, the XGBoost model, and the ensemble average of
their predicted probabilities — the same ensemble predict.py uses in
production — so the ensembling choice is itself measured, not assumed.

Usage (from project root, with venv activated):
    python -m src.ml.evaluate_model
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.ml.feature_engineering import (
    FEATURE_COLUMNS,
    INT_TO_LABEL,
    LABEL_TO_INT,
    build_feature_dataset,
    build_training_rows,
)
from src.ml.train_model import (
    MODELS_DIR,
    ModelNotTrainedError,
    TrainingDataError,
    load_trained_models,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

EVALUATION_REPORT_PATH = MODELS_DIR / "evaluation_report.json"

CLASS_LABELS_IN_ORDER = [INT_TO_LABEL[i] for i in range(len(INT_TO_LABEL))]


def rebuild_test_set(metadata: dict) -> pd.DataFrame:
    """
    Rebuild the exact same held-out test set train_model.py evaluated
    against, using the test_cutoff_date recorded in its metadata.

    Args:
        metadata: Output of load_trained_models()'s metadata element.

    Returns:
        A DataFrame of usable rows (complete features, non-null label)
        with date strictly after metadata["test_cutoff_date"].

    Raises:
        TrainingDataError: If no rows fall after the cutoff date — e.g.
            no data has been loaded since training, so there's nothing
            new to evaluate against.
    """
    dataset = build_feature_dataset(metadata.get("train_companies"))
    train_rows = build_training_rows(dataset)

    cutoff_date = pd.Timestamp(metadata["test_cutoff_date"]).date()
    test_df = train_rows[train_rows["date"] > cutoff_date].reset_index(drop=True)

    if test_df.empty:
        raise TrainingDataError(
            f"No rows found after test_cutoff_date ({cutoff_date}) — nothing to "
            f"evaluate. This is expected if no new data has been loaded since "
            f"the last training run."
        )

    return test_df


def evaluate_predictions(y_true, y_pred, model_name: str) -> dict:
    """
    Compute and log accuracy, a full classification report, and a
    confusion matrix for one model's predictions.

    Args:
        y_true: Integer-encoded true labels.
        y_pred: Integer-encoded predicted labels.
        model_name: Human-readable name for logging (e.g. "RandomForest").

    Returns:
        A dict with accuracy, a per-class precision/recall/f1 report
        (from sklearn's classification_report, output_dict=True), and
        the confusion matrix as a nested list (rows = true class,
        columns = predicted class, in CLASS_LABELS_IN_ORDER order).
    """
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(CLASS_LABELS_IN_ORDER))),
        target_names=CLASS_LABELS_IN_ORDER,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_LABELS_IN_ORDER))))

    logger.info("--- %s ---", model_name)
    logger.info("Accuracy: %.4f", accuracy)
    logger.info(
        "Macro F1: %.4f | Macro Precision: %.4f | Macro Recall: %.4f",
        report["macro avg"]["f1-score"],
        report["macro avg"]["precision"],
        report["macro avg"]["recall"],
    )
    logger.info("Confusion matrix (rows=actual, cols=predicted, order=%s):\n%s", CLASS_LABELS_IN_ORDER, matrix)

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }


def top_feature_importances(model, top_n: int = 5) -> list[tuple[str, float]]:
    """
    Return a model's top-N most important features.

    Args:
        model: A fitted model exposing a feature_importances_ attribute
            (true of both RandomForestClassifier and XGBClassifier).
        top_n: How many features to return.

    Returns:
        A list of (feature_name, importance) tuples, sorted descending
        by importance.
    """
    importances = model.feature_importances_
    ranked = sorted(
        ((name, float(importance)) for name, importance in zip(FEATURE_COLUMNS, importances)),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return ranked[:top_n]


def run_evaluation() -> dict:
    """
    End-to-end evaluation run: load the trained models, rebuild the
    held-out test set, and evaluate RandomForest, XGBoost, and their
    ensemble average — saving a combined report to models/evaluation_report.json.

    Returns:
        The evaluation report dict that was written to disk, with keys
        "random_forest", "xgboost", and "ensemble", each holding the
        output of evaluate_predictions() plus (for the two individual
        models) top feature importances.
    """
    rf_model, xgb_model, metadata = load_trained_models()
    test_df = rebuild_test_set(metadata)

    logger.info("Evaluating on %d held-out row(s) (test_cutoff_date=%s).",
                len(test_df), metadata["test_cutoff_date"])

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["label"].map(LABEL_TO_INT)

    rf_proba = rf_model.predict_proba(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)
    ensemble_proba = (rf_proba + xgb_proba) / 2.0

    rf_report = evaluate_predictions(y_test, np.argmax(rf_proba, axis=1), "RandomForest")
    rf_report["top_features"] = top_feature_importances(rf_model)

    xgb_report = evaluate_predictions(y_test, np.argmax(xgb_proba, axis=1), "XGBoost")
    xgb_report["top_features"] = top_feature_importances(xgb_model)

    ensemble_report = evaluate_predictions(y_test, np.argmax(ensemble_proba, axis=1), "Ensemble (RF + XGBoost avg)")

    full_report = {
        "test_cutoff_date": metadata["test_cutoff_date"],
        "test_rows": len(test_df),
        "random_forest": rf_report,
        "xgboost": xgb_report,
        "ensemble": ensemble_report,
    }

    EVALUATION_REPORT_PATH.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    logger.info("Saved evaluation report to %s", EVALUATION_REPORT_PATH)

    return full_report


def main() -> None:
    """Entry point for standalone script execution."""
    try:
        run_evaluation()
    except (ModelNotTrainedError, TrainingDataError) as exc:
        logger.error("Evaluation aborted: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
