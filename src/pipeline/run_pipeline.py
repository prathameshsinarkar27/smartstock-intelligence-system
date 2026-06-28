"""
run_pipeline.py

Single-entry-point for the SmartStock Intelligence Platform's
data pipeline. Runs the complete data collection + ETL workflow — ingestion
(stock prices, company fundamentals, news) followed by ETL (clean,
transform, load) — with one command:

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
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class StageResult:
    """
    Outcome of a single pipeline stage, used to build the end-of-run
    summary.

    Attributes:
        name: Human-readable stage name, e.g. "Fetch stock prices (AAPL)".
        success: Whether the stage completed without error.
        duration_seconds: How long the stage took to run.
        detail: Optional extra context (e.g. an error message) for the summary.
    """

    name: str
    success: bool
    duration_seconds: float
    detail: str = ""


@dataclass
class PipelineRunReport:
    """
    Aggregated outcome of an entire pipeline run, used to decide the
    process exit code and to print the final summary.

    Attributes:
        stage_results: One StageResult per stage that was attempted.
        critical_failure: Set if a pipeline-critical stage failed, which
            stops the run early regardless of remaining symbols.
    """

    stage_results: list[StageResult] = field(default_factory=list)
    critical_failure: bool = False

    @property
    def all_succeeded(self) -> bool:
        """True if every recorded stage succeeded and no critical failure occurred."""
        return not self.critical_failure and all(r.success for r in self.stage_results)


def _run_stage(report: PipelineRunReport, stage_name: str, stage_fn, *args, result_check=None, **kwargs):
    """
    Run a single pipeline stage with consistent start/end/timing logging
    and error capture, recording the outcome on the shared report.

    Args:
        report: The PipelineRunReport to append this stage's result to.
        stage_name: Human-readable name for logging, e.g. "Fetch news (AAPL)".
        stage_fn: The existing ingestion/ETL function to call.
        *args: Positional arguments forwarded to stage_fn.
        result_check: Optional callable taking stage_fn's return value and
            returning True if the stage should be considered successful.
            Needed because some existing ETL functions (e.g.
            run_transform_for_symbol, run_load_for_symbol) catch their own
            per-data-type errors internally and return a partial dict
            instead of raising — so the mere absence of an exception does
            not guarantee the work this stage cares about actually
            happened. When omitted, any non-exception return counts as
            success (correct for stages that always raise on failure,
            e.g. fetch_and_save_companies).
        **kwargs: Keyword arguments forwarded to stage_fn.

    Returns:
        Whatever stage_fn returns, if the stage succeeded (per
        result_check, when provided). None if the stage raised, or if
        result_check rejected the return value.

    Raises:
        Nothing — all exceptions from stage_fn are caught, logged, and
        recorded in the report as a failed stage. Callers decide whether a
        failed stage should stop the pipeline (pipeline-critical stages)
        or simply be skipped (per-symbol stages).
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
    Run the company data stages: fetch (ingestion) -> transform -> load.

    This is pipeline-critical: historical_prices and news_articles both
    have a NOT NULL foreign key to companies (database/tables.sql), so no
    symbol's price/news data can be loaded if this stage fails.

    Args:
        report: The shared PipelineRunReport to record stage outcomes on.
        symbols: List of stock ticker symbols whose company data should be
            fetched, e.g. ["AAPL", "MSFT"].

    Returns:
        True if all three company sub-stages succeeded, False otherwise.
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
    Run the price and news stages for a single symbol: fetch -> transform
    -> load, for both data types.

    Each sub-stage is isolated: if fetching/transforming/loading prices
    fails for this symbol, news is still attempted, and vice versa. If
    every sub-stage for this symbol fails, the symbol is simply skipped —
    the pipeline continues on to the next symbol rather than stopping.

    The transform and load stages use result_check because
    run_transform_for_symbol()/run_load_for_symbol() (src/etl/) catch
    their own per-data-type errors internally and return a partial dict
    rather than raising (see their docstrings: "a key is omitted if that
    data type failed"). Without checking for the "prices" key explicitly,
    a silent partial failure inside those calls would be misreported as a
    successful stage here.

    Args:
        report: The shared PipelineRunReport to record stage outcomes on.
        symbol: Stock ticker symbol to process, e.g. "AAPL".
    """
    # --- Prices: fetch -> transform -> load ---
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

    # --- News: fetch -> (clean+transform happens together with prices'
    #     run_transform_for_symbol call above, since that function handles
    #     both data types for a symbol in one pass — see
    #     src/etl/transform_data.py:run_transform_for_symbol) ---
    # NOTE: fetch_and_save_news is called here as its own logged stage so
    # the blueprint's "fetch financial news" step is independently visible
    # and independently fault-isolated, even though its cleaning/transform
    # is already covered by the single run_transform_for_symbol() call above.
    _run_stage(report, f"Fetch financial news ({symbol})", fetch_and_save_news, [symbol])


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for standalone script execution."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete SmartStock data pipeline: ingestion (prices, "
            "company data, news) followed by ETL (clean, transform, load)."
        )
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="One or more stock ticker symbols, e.g. --symbols AAPL MSFT GOOGL",
    )
    return parser.parse_args()


def _print_summary(report: PipelineRunReport, total_duration: float) -> None:
    """
    Log a final, human-readable summary of every stage's outcome and the
    total pipeline execution time.

    Args:
        report: The completed PipelineRunReport.
        total_duration: Total wall-clock time for the entire pipeline run, in seconds.
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
    Run the full pipeline for a list of symbols: company data first
    (pipeline-critical), then price + news stages per symbol (fault-isolated
    per symbol).

    Args:
        symbols: List of stock ticker symbols to process, e.g. ["AAPL", "MSFT"].

    Returns:
        A PipelineRunReport describing every stage that ran and whether the
        overall run succeeded.
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
    Entry point for standalone script execution.

    Exit codes:
        0 - every stage succeeded.
        1 - the pipeline-critical company stage failed (no symbols processed).
        2 - the company stage succeeded but at least one per-symbol stage failed.
    """
    if not settings.finnhub_api_key or not settings.twelvedata_api_key or not settings.newsapi_api_key:
        logger.error(
            "One or more required API keys are not set (FINNHUB_API_KEY, "
            "TWELVEDATA_API_KEY, NEWSAPI_API_KEY). Add them to your .env file. "
            "See docs/PHASE_1_SETUP_GUIDE.md for instructions."
        )
        sys.exit(1)

    args = parse_args()
    report = run_pipeline(args.symbols)

    if report.critical_failure:
        sys.exit(1)
    elif not report.all_succeeded:
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
