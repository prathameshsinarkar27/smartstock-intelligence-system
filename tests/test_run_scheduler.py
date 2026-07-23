"""
test_run_scheduler.py

Tests for src/scheduler/run_scheduler.py: HH:MM/weekday parsing and
validation, and that build_scheduler() registers exactly the two jobs
(daily, weekly) with the configured trigger times. Does not start the
scheduler (BlockingScheduler.start() blocks forever by design) — only
construction and job registration are tested.
"""

import os
from dataclasses import replace
from unittest.mock import patch

os.environ.setdefault("FINNHUB_API_KEY", "x")
os.environ.setdefault("NEWSAPI_API_KEY", "x")
os.environ.setdefault("TWELVEDATA_API_KEY", "x")
os.environ.setdefault("GEMINI_API_KEY", "x")

import pytest

from src.scheduler import run_scheduler


def test_parse_hh_mm_valid():
    assert run_scheduler._parse_hh_mm("18:00", "X") == (18, 0)
    assert run_scheduler._parse_hh_mm("09:05", "X") == (9, 5)
    assert run_scheduler._parse_hh_mm("23:59", "X") == (23, 59)


@pytest.mark.parametrize("bad_value", ["18", "18:00:00", "25:00", "18:60", "abc:def", ""])
def test_parse_hh_mm_rejects_invalid(bad_value):
    with pytest.raises(ValueError):
        run_scheduler._parse_hh_mm(bad_value, "SCHEDULER_DAILY_TIME")


def test_validate_weekday_accepts_known_names():
    for day in ("mon", "tue", "wed", "thu", "fri", "sat", "sun"):
        assert run_scheduler._validate_weekday(day, "X") == day


def test_validate_weekday_rejects_unknown():
    with pytest.raises(ValueError, match="SCHEDULER_WEEKLY_DAY"):
        run_scheduler._validate_weekday("someday", "SCHEDULER_WEEKLY_DAY")


def test_build_scheduler_registers_daily_and_weekly_jobs():
    custom_settings = replace(
        run_scheduler.settings,
        scheduler_daily_time="18:00",
        scheduler_weekly_day="sun",
        scheduler_weekly_time="19:30",
        scheduler_timezone="UTC",
    )
    with patch("src.scheduler.run_scheduler.settings", custom_settings):
        scheduler = run_scheduler.build_scheduler()

    job_ids = {job.id for job in scheduler.get_jobs()}
    assert job_ids == {"daily_job", "weekly_job"}

    daily_job = scheduler.get_job("daily_job")
    weekly_job = scheduler.get_job("weekly_job")
    assert "18:00:00" in str(daily_job.trigger) or "hour='18'" in str(daily_job.trigger)
    assert "19:30:00" in str(weekly_job.trigger) or "hour='19'" in str(weekly_job.trigger)


def test_build_scheduler_raises_clear_error_on_bad_config():
    custom_settings = replace(run_scheduler.settings, scheduler_daily_time="not-a-time")
    with patch("src.scheduler.run_scheduler.settings", custom_settings):
        with pytest.raises(ValueError, match="SCHEDULER_DAILY_TIME"):
            run_scheduler.build_scheduler()
