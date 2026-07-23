"""
test_scheduler_jobs.py

Tests for src/scheduler/jobs.py: step-level error isolation (one failed
step doesn't stop the rest), job-level pass/fail reporting, and symbol
resolution fallback. Every underlying pipeline/ML function is mocked —
this module tests orchestration, not the pipeline/ML logic itself
(already covered by earlier phases' own tests).
"""

import os
from unittest.mock import MagicMock, patch

os.environ.setdefault("FINNHUB_API_KEY", "x")
os.environ.setdefault("NEWSAPI_API_KEY", "x")
os.environ.setdefault("TWELVEDATA_API_KEY", "x")
os.environ.setdefault("GEMINI_API_KEY", "x")

from src.scheduler import jobs
from src.utils.config import TrackedSymbolsError


def test_run_step_records_success():
    report = jobs.JobReport(job_name="test")
    result = jobs._run_step(report, "a step", lambda: "ok")

    assert result == "ok"
    assert len(report.steps) == 1
    assert report.steps[0].success is True
    assert report.steps[0].name == "a step"


def test_run_step_records_failure_without_raising():
    report = jobs.JobReport(job_name="test")
    result = jobs._run_step(report, "a step", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    assert result is None
    assert len(report.steps) == 1
    assert report.steps[0].success is False
    assert "boom" in report.steps[0].detail


def test_job_report_all_succeeded():
    report = jobs.JobReport(job_name="test")
    report.steps.append(jobs.JobStepResult(name="a", success=True))
    report.steps.append(jobs.JobStepResult(name="b", success=True))
    assert report.all_succeeded is True

    report.steps.append(jobs.JobStepResult(name="c", success=False))
    assert report.all_succeeded is False


def test_run_daily_job_runs_all_three_steps_in_order():
    call_order = []

    with patch("src.scheduler.jobs.run_pipeline", side_effect=lambda syms: call_order.append("pipeline")) as mock_pipeline, \
         patch("src.scheduler.jobs.run_sentiment_pipeline", side_effect=lambda symbols: call_order.append("sentiment")) as mock_sentiment, \
         patch("src.scheduler.jobs.run_predictions", side_effect=lambda symbols: call_order.append("predict")) as mock_predict:
        report = jobs.run_daily_job(symbols=["AAPL", "MSFT"])

    assert call_order == ["pipeline", "sentiment", "predict"]
    assert report.all_succeeded is True
    assert len(report.steps) == 3
    mock_pipeline.assert_called_once_with(["AAPL", "MSFT"])
    mock_sentiment.assert_called_once_with(symbols=["AAPL", "MSFT"])
    mock_predict.assert_called_once_with(symbols=["AAPL", "MSFT"])


def test_run_daily_job_continues_after_a_failed_step():
    """A failed ingestion step should not prevent sentiment scoring / predictions from still being attempted."""
    with patch("src.scheduler.jobs.run_pipeline", side_effect=RuntimeError("API down")), \
         patch("src.scheduler.jobs.run_sentiment_pipeline", return_value=5) as mock_sentiment, \
         patch("src.scheduler.jobs.run_predictions", return_value=3) as mock_predict:
        report = jobs.run_daily_job(symbols=["AAPL"])

    assert report.all_succeeded is False
    assert len(report.steps) == 3
    assert report.steps[0].success is False
    assert report.steps[1].success is True
    assert report.steps[2].success is True
    mock_sentiment.assert_called_once()
    mock_predict.assert_called_once()


def test_run_daily_job_aborts_cleanly_with_no_symbols():
    with patch("src.scheduler.jobs.load_tracked_symbols", side_effect=TrackedSymbolsError("missing file")), \
         patch("src.scheduler.jobs.run_pipeline") as mock_pipeline:
        report = jobs.run_daily_job(symbols=None)

    assert report.all_succeeded is False
    assert len(report.steps) == 1
    assert "symbols" in report.steps[0].name
    mock_pipeline.assert_not_called()


def test_run_daily_job_uses_tracked_symbols_when_none_given():
    with patch("src.scheduler.jobs.load_tracked_symbols", return_value=["AAPL", "TSLA"]), \
         patch("src.scheduler.jobs.run_pipeline") as mock_pipeline, \
         patch("src.scheduler.jobs.run_sentiment_pipeline"), \
         patch("src.scheduler.jobs.run_predictions"):
        jobs.run_daily_job(symbols=None)

    mock_pipeline.assert_called_once_with(["AAPL", "TSLA"])


def test_run_weekly_job_runs_all_three_steps_on_success():
    with patch("src.scheduler.jobs.run_training", return_value={"random_forest": {}, "xgboost": {}}) as mock_train, \
         patch("src.scheduler.jobs.run_evaluation", return_value={}) as mock_eval, \
         patch("src.scheduler.jobs.run_predictions", return_value=10) as mock_predict:
        report = jobs.run_weekly_job(symbols=["AAPL"])

    assert report.all_succeeded is True
    assert len(report.steps) == 3
    mock_train.assert_called_once_with(["AAPL"])
    mock_eval.assert_called_once()
    mock_predict.assert_called_once_with(symbols=["AAPL"])


def test_run_weekly_job_skips_evaluation_and_prediction_if_training_fails():
    with patch("src.scheduler.jobs.run_training", side_effect=RuntimeError("no usable training rows")), \
         patch("src.scheduler.jobs.run_evaluation") as mock_eval, \
         patch("src.scheduler.jobs.run_predictions") as mock_predict:
        report = jobs.run_weekly_job(symbols=["AAPL"])

    assert report.all_succeeded is False
    assert len(report.steps) == 1
    assert report.steps[0].name == "ML model retraining"
    mock_eval.assert_not_called()
    mock_predict.assert_not_called()


def test_weekly_job_does_not_require_tracked_symbols():
    """Unlike the daily job, a missing tracked_symbols.txt shouldn't abort the weekly job (training can use all DB companies)."""
    with patch("src.scheduler.jobs.load_tracked_symbols", side_effect=TrackedSymbolsError("missing file")), \
         patch("src.scheduler.jobs.run_training", return_value={}) as mock_train, \
         patch("src.scheduler.jobs.run_evaluation", return_value={}), \
         patch("src.scheduler.jobs.run_predictions", return_value=0):
        report = jobs.run_weekly_job(symbols=None)

    mock_train.assert_called_once_with(None)
    assert report.steps[0].success is True
