"""
run_scheduler.py

A long-running process that calls src/scheduler/jobs.py's run_daily_job()
and run_weekly_job() on a recurring schedule, using APScheduler's
BlockingScheduler with cron-style triggers.

Schedule times are configurable via environment variables (read through
src.utils.config.settings, same pattern as Phase 11's RAG rate-limit
settings) rather than hardcoded, so the same Docker image can run on a
different schedule per deployment without a code change:

    SCHEDULER_DAILY_TIME=18:00        # HH:MM, 24h
    SCHEDULER_WEEKLY_DAY=sun          # mon, tue, wed, thu, fri, sat, sun
    SCHEDULER_WEEKLY_TIME=19:00       # HH:MM, 24h
    SCHEDULER_TIMEZONE=UTC            # any IANA timezone name
    SCHEDULER_RUN_ON_STARTUP=false    # if true, also run the daily job
                                      # once immediately on startup —
                                      # useful the first time a fresh
                                      # container starts with an empty
                                      # database, so you're not waiting
                                      # until the next scheduled time to
                                      # see any data.

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
    Parse a "HH:MM" string into (hour, minute), raising a clear error
    (rather than a confusing one from APScheduler/CronTrigger) if the
    configured value is malformed.

    Args:
        value: The "HH:MM" string, e.g. "18:00".
        setting_name: Which setting this came from, for the error
            message (e.g. "SCHEDULER_DAILY_TIME").

    Returns:
        (hour, minute) as integers.

    Raises:
        ValueError: If `value` isn't a valid "HH:MM" 24-hour time.
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
    """Confirm a configured weekday name is one CronTrigger accepts, with a clear error if not."""
    if value not in _WEEKDAY_NAMES:
        raise ValueError(f"{setting_name}={value!r} must be one of {sorted(_WEEKDAY_NAMES)}.")
    return value


def _run_daily_job_job() -> None:
    """APScheduler job wrapper — run_daily_job() already catches its own step errors, so this never raises."""
    logger.info("Scheduled trigger fired: daily job.")
    run_daily_job()


def _run_weekly_job_job() -> None:
    """APScheduler job wrapper — run_weekly_job() already catches its own step errors, so this never raises."""
    logger.info("Scheduled trigger fired: weekly job.")
    run_weekly_job()


def build_scheduler() -> BlockingScheduler:
    """
    Construct (but don't start) a BlockingScheduler with the daily and
    weekly jobs registered per src.utils.config.settings' scheduler_*
    fields.

    Returns:
        A configured BlockingScheduler, not yet started.

    Raises:
        ValueError: If any SCHEDULER_* setting is malformed (caught and
            logged clearly by main() before the process exits, rather
            than surfacing as an opaque APScheduler stack trace).
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
    Entry point: build the scheduler, optionally run the daily job once
    immediately (SCHEDULER_RUN_ON_STARTUP=true), then block forever
    running jobs at their configured times until interrupted
    (SIGINT/SIGTERM — including `docker stop`, which BlockingScheduler
    handles gracefully via its own signal handling).
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
