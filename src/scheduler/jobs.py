"""
jobs.py

Defines the daily and weekly scheduled jobs for SmartStock.

Daily: ingestion/ETL, sentiment scoring, and predictions using the current model.
Weekly: model retraining, evaluation, and predictions using the fresh model.

Both jobs reuse existing pipeline functions and expose CLI subcommands.

Usage:
    python -m src.scheduler.jobs daily
    python -m src.scheduler.jobs weekly
    python -m src.scheduler.jobs daily --symbols AAPL MSFT
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from src.ml.evaluate_model import run_evaluation
from src.ml.predict import run_predictions
from src.ml.train_model import run_training
from src.pipeline.run_pipeline import run_pipeline
from src.sentiment.sentiment_pipeline import run_sentiment_pipeline
from src.utils.config import TrackedSymbolsError, load_tracked_symbols
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class JobStepResult:
    """Outcome of one job step."""

    name: str
    success: bool
    detail: str = ""
    duration_seconds: float = 0.0


@dataclass
class JobReport:
    """Aggregated results for a daily or weekly job."""

    job_name: str
    steps: list[JobStepResult] = field(default_factory=list)
    total_duration_seconds: float = 0.0

    @property
    def all_succeeded(self) -> bool:
        return all(step.success for step in self.steps)


def _run_step(report: JobReport, step_name: str, step_fn: Callable[[], Any]) -> Any | None:
    """
    Run one job step and record its result.

    Args:
        report: Report receiving the step outcome.
        step_name: Human-readable step name.
        step_fn: Zero-argument function performing the step.

    Returns:
        The step result, or None if it raised an exception.
    """
    logger.info("[%s] Starting step: %s", report.job_name, step_name)
    start = time.monotonic()

    try:
        result = step_fn()
    except Exception as exc:
        duration = time.monotonic() - start
        logger.error("[%s] Step failed: %s (%.1fs) — %s", report.job_name, step_name, duration, exc)
        report.steps.append(JobStepResult(name=step_name, success=False, detail=str(exc), duration_seconds=duration))
        return None

    duration = time.monotonic() - start
    logger.info("[%s] Step succeeded: %s (%.1fs)", report.job_name, step_name, duration)
    report.steps.append(JobStepResult(name=step_name, success=True, detail=str(result), duration_seconds=duration))
    return result


def _resolve_symbols(symbols: list[str] | None) -> list[str] | None:
    """
    Resolve symbols from the CLI or tracked-symbols configuration.

    Args:
        symbols: Explicit symbols, or None to use the configured list.

    Returns:
        Resolved symbols, or None if the configured list is unavailable.
    """
    if symbols:
        return symbols

    try:
        return load_tracked_symbols()
    except TrackedSymbolsError as exc:
        logger.warning("Could not load config/tracked_symbols.txt: %s", exc)
        return None


def run_daily_job(symbols: list[str] | None = None) -> JobReport:
    """
    Run ingestion/ETL, sentiment scoring, and current-model predictions.

    Args:
        symbols: Optional symbols to process. Defaults to the tracked-symbols file.

    Returns:
        Job report containing the outcome of each attempted step.
    """
    report = JobReport(job_name="daily")
    overall_start = time.monotonic()

    resolved_symbols = _resolve_symbols(symbols)
    if not resolved_symbols:
        report.steps.append(JobStepResult(
            name="resolve symbols", success=False,
            detail="No symbols provided and config/tracked_symbols.txt could not be read.",
        ))
        report.total_duration_seconds = time.monotonic() - overall_start
        logger.error("[daily] Aborting — no symbols to run against.")
        return report

    _run_step(report, "ingestion + ETL", lambda: run_pipeline(resolved_symbols))
    _run_step(report, "sentiment scoring", lambda: run_sentiment_pipeline(symbols=resolved_symbols))
    _run_step(report, "ML predictions (current model)", lambda: run_predictions(symbols=resolved_symbols))

    report.total_duration_seconds = time.monotonic() - overall_start
    _log_summary(report)
    return report


def run_weekly_job(symbols: list[str] | None = None) -> JobReport:
    """
    Retrain, evaluate, and generate predictions with the new models.

    Args:
        symbols: Optional symbols to restrict training and prediction.

    Returns:
        Job report containing the outcome of each attempted step.
    """
    report = JobReport(job_name="weekly")
    overall_start = time.monotonic()

    resolved_symbols = _resolve_symbols(symbols)

    training_result = _run_step(report, "ML model retraining", lambda: run_training(resolved_symbols))
    if training_result is not None:
        _run_step(report, "ML model evaluation", run_evaluation)
        _run_step(report, "ML predictions (fresh model)", lambda: run_predictions(symbols=resolved_symbols))
    else:
        logger.warning("[weekly] Skipping evaluation and prediction — retraining did not succeed.")

    report.total_duration_seconds = time.monotonic() - overall_start
    _log_summary(report)
    return report


def _log_summary(report: JobReport) -> None:
    """Log a compact pass/fail summary for a completed job, mirroring run_pipeline.py's own summary style."""
    logger.info("=" * 60)
    logger.info("JOB SUMMARY: %s (%.1fs total)", report.job_name, report.total_duration_seconds)
    for step in report.steps:
        status = "OK" if step.success else "FAILED"
        logger.info("  [%s] %s", status, step.name)
    logger.info("JOB RESULT: %s", "SUCCESS" if report.all_succeeded else "COMPLETED WITH FAILURES")
    logger.info("=" * 60)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the SmartStock scheduler's daily or weekly job once and exit."
    )
    parser.add_argument("job", choices=["daily", "weekly"], help="Which job to run.")
    parser.add_argument(
        "--symbols", nargs="+", metavar="SYMBOL",
        help="Restrict to specific symbols instead of config/tracked_symbols.txt "
        "(weekly job: instead of every company in the database).",
    )
    return parser.parse_args()


def main() -> None:
    """
    Run the selected scheduled job.

    Exits with status 0 when all steps succeed, otherwise 1.
    """
    args = parse_args()

    if args.job == "daily":
        report = run_daily_job(symbols=args.symbols)
    else:
        report = run_weekly_job(symbols=args.symbols)

    sys.exit(0 if report.all_succeeded else 1)


if __name__ == "__main__":
    main()
