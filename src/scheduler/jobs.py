"""
jobs.py

The two recurring jobs the automatic scheduler runs (Phase 14), each a
thin orchestration wrapper around functions that already existed as
standalone scripts since earlier phases — no pipeline logic is
duplicated here, only sequencing and unified logging/error handling so a
failure in one step doesn't silently stop the others or crash whatever
is calling this (a Docker scheduler loop or a Windows Task Scheduler
task).

Daily job (data freshness):
    1. Ingestion + ETL for every tracked symbol (Phases 1-3)
    2. Sentiment scoring of any newly-ingested articles (Phase 7)
    3. ML predictions using the CURRENTLY SAVED model (Phase 8) — this
       does NOT retrain; it scores today's data against whatever model
       models/ already has, which is fast and appropriate to run daily.

Weekly job (model freshness):
    1. Retrain both models on the latest available data (Phase 8)
    2. Evaluate the newly-trained models and save the report (Phase 8)
    3. Re-run predictions so the dashboard immediately reflects the
       freshly retrained model rather than waiting for tomorrow's daily
       job

Splitting retraining out from the daily job is deliberate: retraining
rebuilds the entire feature dataset from scratch and is the most
expensive step in the whole pipeline (see train_model.py's docstring) —
running it daily would be wasteful for a project-scale symbol list where
the underlying patterns don't meaningfully shift day to day. Both jobs
are independent CLI subcommands so either can also be run manually.

Usage (from project root, with venv activated):
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
    """Outcome of one step within a job (e.g. "ingestion", "sentiment scoring")."""

    name: str
    success: bool
    detail: str = ""
    duration_seconds: float = 0.0


@dataclass
class JobReport:
    """
    Outcome of a full job run (daily or weekly), covering every step
    attempted regardless of whether earlier steps failed.
    """

    job_name: str
    steps: list[JobStepResult] = field(default_factory=list)
    total_duration_seconds: float = 0.0

    @property
    def all_succeeded(self) -> bool:
        return all(step.success for step in self.steps)


def _run_step(report: JobReport, step_name: str, step_fn: Callable[[], Any]) -> Any | None:
    """
    Run one job step, catching any exception so it's recorded as a
    failed step rather than propagating and aborting the remaining
    steps (or, for run_scheduler.py's long-running loop, the whole
    process).

    Args:
        report: The JobReport to append this step's outcome to.
        step_name: Human-readable label used in logs and the report.
        step_fn: A zero-argument callable performing the step.

    Returns:
        Whatever step_fn() returned, or None if it raised.
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
    Resolve the symbol list a job should run against: an explicit
    override if given, otherwise config/tracked_symbols.txt.

    Args:
        symbols: An explicit symbol list (e.g. from --symbols on the
            CLI), or None to fall back to the tracked-symbols file.

    Returns:
        The resolved symbol list, or None if neither an override nor a
        readable tracked_symbols.txt is available — callers that can
        operate over "all companies already in the database" (ML
        training/prediction) treat None as that; run_daily_job() cannot
        (ingestion needs an explicit symbol list) and raises instead.
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
    Run the daily job: ingestion + ETL, sentiment scoring, and
    predictions using the current model — for every tracked symbol.

    Args:
        symbols: Explicit symbol list to restrict the run to. If None,
            uses config/tracked_symbols.txt.

    Returns:
        A JobReport with one JobStepResult per step attempted. A failed
        or missing symbol list still produces a report (with a single
        failed "resolve symbols" step) rather than raising, so this is
        always safe to call from a long-running scheduler loop.
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
    Run the weekly job: retrain both ML models on the latest data,
    evaluate them, and refresh predictions with the newly-trained model.

    Args:
        symbols: Explicit symbol list to restrict training/prediction
            to. If None, every company with enough history in the
            database is used (see train_model.run_training()'s
            docstring) — unlike run_daily_job(), a missing
            tracked_symbols.txt is not fatal here, since training can
            fall back to "everything in the database."

    Returns:
        A JobReport with one JobStepResult per step attempted.
        "ML predictions (fresh model)" is skipped (not attempted) if
        retraining itself failed, since there would be no new model to
        predict with — this is reflected as a step never being added to
        the report, not as a spurious failed step.
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
    """Parse command-line arguments for standalone script execution."""
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
    Entry point for standalone script execution
    (`python -m src.scheduler.jobs daily|weekly`), used directly by the
    Windows Task Scheduler wrapper scripts in scripts/windows/, and
    indirectly (via run_scheduler.py) by the Docker `scheduler` service.

    Exits with status 0 if every step succeeded, 1 otherwise, so a
    Windows Task Scheduler task (or any other caller checking the
    process exit code) can detect a failed run without parsing log
    output. run_daily_job()/run_weekly_job() never raise — every step
    they run is wrapped by _run_step(), which converts an exception into
    a failed JobStepResult — so no exception handling is needed here.
    """
    args = parse_args()

    if args.job == "daily":
        report = run_daily_job(symbols=args.symbols)
    else:
        report = run_weekly_job(symbols=args.symbols)

    sys.exit(0 if report.all_succeeded else 1)


if __name__ == "__main__":
    main()
