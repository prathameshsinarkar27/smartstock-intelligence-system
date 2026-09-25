"""
run_scheduler.py

Runs daily and weekly SmartStock jobs on a recurring APScheduler schedule.

Schedule settings are loaded from environment-backed configuration:
    SCHEDULER_DAILY_TIME=18:00
    SCHEDULER_WEEKLY_DAY=sun
    SCHEDULER_WEEKLY_TIME=19:00
    SCHEDULER_TIMEZONE=UTC
    SCHEDULER_RUN_ON_STARTUP=false

Usage:
    python -m src.scheduler.run_scheduler
"""

import os

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from src.scheduler.jobs import run_daily_job, run_weekly_job
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_WEEKDAY_NAMES = {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}


def _parse_hh_mm(value: str, setting_name: str) -> tuple[int, int]:
    """
    Parse a 24-hour HH:MM setting.

    Args:
        value: Configured time string.
        setting_name: Setting name used in validation errors.

    Returns:
        Hour and minute as integers.

    Raises:
        ValueError: If the value is not a valid HH:MM time.
    """
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError(f"{setting_name}={value!r} is not a valid HH:MM time.")

    try:
        hour, minute = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"{setting_name}={value!r} is not a valid HH:MM time.") from exc

    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"{setting_name}={value!r} is out of range (hour 0-23, minute 0-59).")

    return hour, minute


def _validate_weekday(value: str, setting_name: str) -> str:
    """Validate a configured weekday name."""
    if value not in _WEEKDAY_NAMES:
        raise ValueError(f"{setting_name}={value!r} must be one of {sorted(_WEEKDAY_NAMES)}.")
    return value


def _run_daily_job_job() -> None:
    """Run the daily job from the APScheduler trigger."""
    logger.info("Scheduled trigger fired: daily job.")
    run_daily_job()


def _run_weekly_job_job() -> None:
    """Run the weekly job from the APScheduler trigger."""
    logger.info("Scheduled trigger fired: weekly job.")
    run_weekly_job()


def build_scheduler() -> BlockingScheduler:
    """
    Build the scheduler with configured daily and weekly jobs.

    Returns:
        Configured BlockingScheduler, not yet started.

    Raises:
        ValueError: If a scheduler setting is invalid.
    """
    daily_hour, daily_minute = _parse_hh_mm(settings.scheduler_daily_time, "SCHEDULER_DAILY_TIME")
    weekly_day = _validate_weekday(settings.scheduler_weekly_day, "SCHEDULER_WEEKLY_DAY")
    weekly_hour, weekly_minute = _parse_hh_mm(settings.scheduler_weekly_time, "SCHEDULER_WEEKLY_TIME")

    scheduler = BlockingScheduler(timezone=settings.scheduler_timezone)

    scheduler.add_job(
        _run_daily_job_job,
        trigger=CronTrigger(hour=daily_hour, minute=daily_minute, timezone=settings.scheduler_timezone),
        id="daily_job",
        name="Daily: ingestion + ETL + sentiment + predictions",
        misfire_grace_time=3600,
        coalesce=True,
    )
    scheduler.add_job(
        _run_weekly_job_job,
        trigger=CronTrigger(
            day_of_week=weekly_day, hour=weekly_hour, minute=weekly_minute, timezone=settings.scheduler_timezone,
        ),
        id="weekly_job",
        name="Weekly: model retraining + evaluation + predictions",
        misfire_grace_time=3600,
        coalesce=True,
    )

    logger.info(
        "Scheduled daily job for %02d:%02d %s; weekly job for %s %02d:%02d %s.",
        daily_hour, daily_minute, settings.scheduler_timezone,
        weekly_day, weekly_hour, weekly_minute, settings.scheduler_timezone,
    )

    return scheduler


def main() -> None:
    """
    Start the scheduler and optionally run the daily job immediately.

    SCHEDULER_RUN_ON_STARTUP=true runs the daily job once before
    entering the schedule loop.
    """
    try:
        scheduler = build_scheduler()
    except ValueError as exc:
        logger.error("Invalid scheduler configuration: %s", exc)
        raise SystemExit(1) from exc

    if os.getenv("SCHEDULER_RUN_ON_STARTUP", "false").strip().lower() in ("1", "true", "yes"):
        logger.info("SCHEDULER_RUN_ON_STARTUP is set — running the daily job once now before entering the schedule loop.")
        run_daily_job()

    logger.info("Scheduler starting. Press Ctrl+C (or `docker stop`) to exit.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
