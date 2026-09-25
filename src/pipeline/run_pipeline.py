"""
run_pipeline.py

Runs the complete SmartStock data pipeline: ingestion, cleaning,
transformation, and loading into PostgreSQL.

    python -m src.pipeline.run_pipeline --symbols AAPL MSFT

Pipeline stages, in order:
    1. Fetch historical stock price data   (src.ingestion.fetch_stock_data)
    2. Fetch company profile/fundamentals  (src.ingestion.fetch_company_data)
    3. Fetch financial news                (src.ingestion.fetch_news)
    4. Clean raw datasets                  (src.etl.clean_stock_data, via transform)
    5. Transform cleaned datasets          (src.etl.transform_data)
    6. Load transformed data into Postgres (src.etl.load_to_db)

"""

import argparse
import sys
import time
from dataclasses import dataclass, field

from src.etl.load_to_db import run_load_companies, run_load_for_symbol
from src.etl.transform_data import run_transform_companies, run_transform_for_symbol
from src.ingestion.fetch_company_data import fetch_and_save_companies
from src.ingestion.fetch_news import fetch_and_save_symbols as fetch_and_save_news
from src.ingestion.fetch_stock_data import fetch_and_save_symbols as fetch_and_save_prices
from src.utils.config import TrackedSymbolsError, load_tracked_symbols, settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class StageResult:
    """Outcome and timing information for a pipeline stage."""

    name: str
    success: bool
    duration_seconds: float
    detail: str = ""


@dataclass
class PipelineRunReport:
    """Aggregated results for a complete pipeline run."""

    stage_results: list[StageResult] = field(default_factory=list)
    critical_failure: bool = False

    @property
    def all_succeeded(self) -> bool:
        """True if every recorded stage succeeded and no critical failure occurred."""
        return not self.critical_failure and all(r.success for r in self.stage_results)


def _run_stage(report: PipelineRunReport, stage_name: str, stage_fn, *args, result_check=None, **kwargs):
    """
    Run a pipeline stage with logging, timing, and error handling.

    Args:
        report: Report receiving the stage result.
        stage_name: Human-readable stage name.
        stage_fn: Function executed for the stage.
        *args: Positional arguments passed to stage_fn.
        result_check: Optional validation function for partial-result stages.
        **kwargs: Keyword arguments passed to stage_fn.

    Returns:
        The stage result, or None if the stage failed.
    """
    logger.info("--- STAGE START: %s ---", stage_name)
    start_time = time.monotonic()

    try:
        result = stage_fn(*args, **kwargs)
        duration = time.monotonic() - start_time

        if result_check is not None and not result_check(result):
            logger.error(
                "--- STAGE FAILED: %s (%.2fs): completed without raising, "
                "but did not produce the expected result (see preceding "
                "error log lines from the called module for details) ---",
                stage_name, duration,
            )
            report.stage_results.append(
                StageResult(stage_name, False, duration, detail="Incomplete result (see logs above)")
            )
            return None

        logger.info("--- STAGE COMPLETE: %s (%.2fs) ---", stage_name, duration)
        report.stage_results.append(StageResult(stage_name, True, duration))
        return result
    except Exception as exc:  # noqa: BLE001 - intentionally broad: any stage failure must be caught, logged, and recorded without crashing the orchestrator
        duration = time.monotonic() - start_time
        logger.error("--- STAGE FAILED: %s (%.2fs): %s ---", stage_name, duration, exc)
        report.stage_results.append(StageResult(stage_name, False, duration, detail=str(exc)))
        return None


def run_company_stage(report: PipelineRunReport, symbols: list[str]) -> bool:
    """
    Run company data stages: fetch, transform, and load.

    This stage is pipeline-critical because price and news records
    depend on company IDs in the database.

    Args:
        report: Report receiving stage results.
        symbols: Stock ticker symbols to process.

    Returns:
        True if all company stages succeed.
    """
    fetch_result = _run_stage(
        report, "Fetch company profile/fundamentals", fetch_and_save_companies, symbols
    )
    if fetch_result is None:
        return False

    transform_result = _run_stage(
        report, "Clean + transform company data", run_transform_companies
    )
    if transform_result is None:
        return False

    load_result = _run_stage(report, "Load company data into PostgreSQL", run_load_companies)
    if load_result is None:
        return False

    return True


def run_symbol_stages(report: PipelineRunReport, symbol: str) -> None:
    """
    Run price and news stages for a single symbol.

    Each stage is isolated so a failure in one data type does not
    prevent processing of the other.

    Args:
        report: Report receiving stage results.
        symbol: Stock ticker symbol to process.
    """
    # Prices: fetch -> transform -> load.
    price_fetch = _run_stage(
        report, f"Fetch stock prices ({symbol})", fetch_and_save_prices, [symbol]
    )
    if price_fetch:
        price_transform = _run_stage(
            report,
            f"Clean + transform prices ({symbol})",
            run_transform_for_symbol,
            symbol,
            result_check=lambda r: "prices" in r,
        )
        if price_transform is not None:
            _run_stage(
                report,
                f"Load prices into PostgreSQL ({symbol})",
                run_load_for_symbol,
                symbol,
                result_check=lambda r: "prices" in r,
            )

    # News is fetched separately, while its transform/load is handled
    # by run_transform_for_symbol() and run_load_for_symbol().
    _run_stage(report, f"Fetch financial news ({symbol})", fetch_and_save_news, [symbol])


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete SmartStock data pipeline: ingestion (prices, "
            "company data, news) followed by ETL (clean, transform, load)."
        )
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        required=False,
        default=None,
        help=(
            "One or more stock ticker symbols, e.g. --symbols AAPL MSFT GOOGL. "
            "If omitted, the pipeline loads its default symbol list from "
            "config/tracked_symbols.txt instead."
        ),
    )
    return parser.parse_args()


def _print_summary(report: PipelineRunReport, total_duration: float) -> None:
    """
    Log the outcome of each stage and total execution time.

    Args:
        report: Completed pipeline report.
        total_duration: Total pipeline runtime in seconds.
    """
    logger.info("=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)

    for stage in report.stage_results:
        status = "OK" if stage.success else "FAILED"
        logger.info("[%-6s] %-45s %6.2fs", status, stage.name, stage.duration_seconds)
        if not stage.success and stage.detail:
            logger.info("           -> %s", stage.detail)

    logger.info("-" * 60)
    logger.info("Total execution time: %.2fs", total_duration)

    if report.all_succeeded:
        logger.info("PIPELINE RESULT: SUCCESS")
    else:
        logger.info("PIPELINE RESULT: COMPLETED WITH FAILURES")
    logger.info("=" * 60)


def run_pipeline(symbols: list[str]) -> PipelineRunReport:
    """
    Run the full pipeline for the provided symbols.

    Company data runs first because price and news records depend on
    company IDs. Price and news stages then run independently per symbol.

    Args:
        symbols: Stock ticker symbols to process.

    Returns:
        Report describing stage outcomes and overall status.
    """
    report = PipelineRunReport()
    pipeline_start = time.monotonic()

    logger.info("=" * 60)
    logger.info("PIPELINE START: symbols=%s", symbols)
    logger.info("=" * 60)

    company_stage_ok = run_company_stage(report, symbols)

    if not company_stage_ok:
        report.critical_failure = True
        total_duration = time.monotonic() - pipeline_start
        logger.error(
            "Company data stage failed. Stopping pipeline — price and news "
            "stages depend on company_id and cannot proceed for any symbol."
        )
        _print_summary(report, total_duration)
        return report

    for symbol in symbols:
        run_symbol_stages(report, symbol)

    total_duration = time.monotonic() - pipeline_start
    _print_summary(report, total_duration)
    return report


def main() -> None:
    """
    Run the pipeline from the command line.

    Uses --symbols when provided; otherwise loads symbols from
    config/tracked_symbols.txt.

    Exit codes:
        0: All stages succeeded.
        1: Critical company stage failed or required configuration is missing.
        2: Company stage succeeded but at least one symbol stage failed.
    """
    if not settings.finnhub_api_key or not settings.twelvedata_api_key or not settings.newsapi_api_key:
        logger.error(
            "One or more required API keys are not set (FINNHUB_API_KEY, "
            "TWELVEDATA_API_KEY, NEWSAPI_API_KEY). Add them to your .env file. "
            "See docs/PHASE_1_SETUP_GUIDE.md for instructions."
        )
        sys.exit(1)

    args = parse_args()

    if args.symbols:
        symbols = args.symbols
        logger.info("Using %d symbol(s) from --symbols: %s", len(symbols), symbols)
    else:
        try:
            symbols = load_tracked_symbols()
        except TrackedSymbolsError as exc:
            logger.error(str(exc))
            sys.exit(1)
        logger.info(
            "No --symbols provided; loaded %d symbol(s) from %s",
            len(symbols), settings.tracked_symbols_path,
        )

    report = run_pipeline(symbols)

    if report.critical_failure:
        sys.exit(1)
    elif not report.all_succeeded:
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
