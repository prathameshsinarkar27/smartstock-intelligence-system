"""
kpi_calculator.py

Computes the KPI metrics shown on the dashboard's Market Overview and
Company Detail pages: current price, daily % change, market cap, trading
volume, P/E ratio, and high/low over the available price history (the
blueprint's "52-Week High/Low" cards, computed over whatever date range is
actually loaded — see the docstring on `get_company_kpis` for why).

"""

from datetime import date
from typing import Any

from src.utils.database import get_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_market_overview_kpis() -> dict[str, Any]:
    """
    Compute market-wide KPI metrics across all tracked companies, for the
    Market Overview page's top KPI row.

    Returns:
        A dict with:
            - total_companies: count of rows in the companies table.
            - total_sectors: count of distinct non-null sectors.
            - advancers_count: companies whose latest close is higher than
              their previous close.
            - decliners_count: companies whose latest close is lower than
              their previous close.
            - avg_daily_change_pct: average daily % change across all
              companies with at least two days of price history.
        All counts are 0 and avg_daily_change_pct is None if no companies
        are loaded yet.
    """
    query = """
        WITH ranked_prices AS (
            SELECT
                company_id,
                close,
                date,
                ROW_NUMBER() OVER (PARTITION BY company_id ORDER BY date DESC) AS rn
            FROM historical_prices
        ),
        latest_two AS (
            SELECT
                company_id,
                MAX(CASE WHEN rn = 1 THEN close END) AS latest_close,
                MAX(CASE WHEN rn = 2 THEN close END) AS previous_close
            FROM ranked_prices
            WHERE rn IN (1, 2)
            GROUP BY company_id
        )
        SELECT
            (SELECT COUNT(*) FROM companies) AS total_companies,
            (SELECT COUNT(DISTINCT sector) FROM companies WHERE sector IS NOT NULL) AS total_sectors,
            COUNT(*) FILTER (WHERE latest_close > previous_close) AS advancers_count,
            COUNT(*) FILTER (WHERE latest_close < previous_close) AS decliners_count,
            AVG((latest_close - previous_close) / previous_close * 100) AS avg_daily_change_pct
        FROM latest_two
        WHERE previous_close IS NOT NULL AND previous_close != 0;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            row = cur.fetchone()

    if row is None:
        return {
            "total_companies": 0,
            "total_sectors": 0,
            "advancers_count": 0,
            "decliners_count": 0,
            "avg_daily_change_pct": None,
        }

    total_companies, total_sectors, advancers_count, decliners_count, avg_daily_change_pct = row
    return {
        "total_companies": total_companies or 0,
        "total_sectors": total_sectors or 0,
        "advancers_count": advancers_count or 0,
        "decliners_count": decliners_count or 0,
        "avg_daily_change_pct": float(avg_daily_change_pct) if avg_daily_change_pct is not None else None,
    }


def _get_company_sentiment_score(conn: Any, company_id: int) -> float | None:
    """
    Compute a single signed sentiment figure for a company's Sentiment
    Score KPI card, from its scored news articles
    (src/sentiment/sentiment_pipeline.py, Phase 7).

    Each scored article contributes its confidence_score with a sign
    matching its label (+confidence for positive, -confidence for
    negative, 0 for neutral), and the figure is the mean of those signed
    values across all scored articles, scaled to a -100..100 range so it
    renders through the same `| percent` template filter used by
    daily_change_pct elsewhere on this page.

    Args:
        conn: An open database connection (reused from the caller's
            get_connection() block rather than opening a second one).
        company_id: The company's surrogate key.

    Returns:
        The signed sentiment figure in [-100, 100], or None if the
        company has no scored news articles yet (distinct from a
        computed 0.0, which means the scored articles average out to
        neutral).
    """
    query = """
        SELECT ss.sentiment, ss.confidence_score
        FROM sentiment_scores ss
        JOIN news_articles na ON na.news_id = ss.news_id
        WHERE na.company_id = %s;
    """

    with conn.cursor() as cur:
        cur.execute(query, (company_id,))
        rows = cur.fetchall()

    if not rows:
        return None

    signed_values = []
    for sentiment, confidence_score in rows:
        confidence = float(confidence_score)
        if sentiment == "positive":
            signed_values.append(confidence)
        elif sentiment == "negative":
            signed_values.append(-confidence)
        else:
            signed_values.append(0.0)

    return (sum(signed_values) / len(signed_values)) * 100


def _get_company_ml_risk_score(conn: Any, company_id: int) -> float | None:
    """
    Fetch a company's most recent ML-predicted risk score for the ML Risk
    Score KPI card, from the predictions table
    (src/ml/predict.py, Phase 8).

    Args:
        conn: An open database connection (reused from the caller's
            get_connection() block).
        company_id: The company's surrogate key.

    Returns:
        The risk_score from that company's row in the latest_predictions
        view (database/views.sql) — already in [0, 1], representing the
        model ensemble's estimated probability of a downward move over
        its prediction horizon (see src/ml/predict.py's module docstring)
        — or None if predict.py hasn't been run for this company yet.
    """
    query = """
        SELECT risk_score
        FROM latest_predictions
        WHERE company_id = %s;
    """

    with conn.cursor() as cur:
        cur.execute(query, (company_id,))
        row = cur.fetchone()

    return float(row[0]) if row is not None else None


def get_company_kpis(symbol: str) -> dict[str, Any] | None:
    """
    Compute the full set of KPI metrics for a single company's Company
    Detail page.

    The blueprint calls for "52-Week High/Low" cards. Computing a true
    52-week figure requires a full year of loaded price history, which may
    not exist yet depending on how much data has been ingested (see
    docs/PHASE_5_TESTING_GUIDE.md). Rather than mislabel a shorter range as
    "52-week," this function computes high/low over whatever date range is
    actually available and returns that range's start/end dates alongside
    the figures, so the template can show an accurate tooltip/subtitle
    (e.g. "High/Low over available data: Apr 20 - Jun 20, 2026") instead of
    a potentially misleading fixed label.

    Args:
        symbol: Stock ticker symbol, e.g. "AAPL".

    Returns:
        None if the symbol has no row in the companies table. Otherwise a
        dict with:
            - symbol, company_name, sector, industry
            - current_price, previous_close, daily_change_pct
            - market_cap, pe_ratio
            - volume (latest day's volume)
            - period_high, period_low: high/low over available price history
            - period_start_date, period_end_date: the actual date range
              period_high/period_low were computed over
            - sentiment_score: signed mean sentiment across this
              company's scored news articles, in [-100, 100] (Phase 7).
              None if no articles have been scored yet.
            - ml_risk_score: the model ensemble's most recent estimated
              probability of a downward price move, in [0, 1] (Phase 8,
              see src/ml/predict.py). None if predict.py hasn't been run
              for this company yet.
            - ai_recommendation: reserved for Phase 10, always None for
              now so templates can render a "Coming in Phase X"
              placeholder without a future template change.
    """
    company_query = """
        SELECT company_id, symbol, company_name, sector, industry, market_cap, pe_ratio
        FROM companies
        WHERE symbol = %s;
    """
    price_history_query = """
        SELECT date, close, volume
        FROM historical_prices
        WHERE company_id = %s
        ORDER BY date DESC;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(company_query, (symbol.upper(),))
            company_row = cur.fetchone()

            if company_row is None:
                return None

            company_id, db_symbol, company_name, sector, industry, market_cap, pe_ratio = company_row

            cur.execute(price_history_query, (company_id,))
            price_rows = cur.fetchall()

        # Computed here (Phase 7) while the connection is still open, since
        # it's independent of price history and needed in both branches
        # below (a company can have scored news with no price data yet,
        # or vice versa).
        sentiment_score = _get_company_sentiment_score(conn, company_id)
        ml_risk_score = _get_company_ml_risk_score(conn, company_id)

    if not price_rows:
        # Company exists but has no price history loaded yet.
        return {
            "symbol": db_symbol,
            "company_name": company_name,
            "sector": sector,
            "industry": industry,
            "current_price": None,
            "previous_close": None,
            "daily_change_pct": None,
            "market_cap": float(market_cap) if market_cap is not None else None,
            "pe_ratio": float(pe_ratio) if pe_ratio is not None else None,
            "volume": None,
            "period_high": None,
            "period_low": None,
            "period_start_date": None,
            "period_end_date": None,
            "sentiment_score": sentiment_score,
            "ml_risk_score": ml_risk_score,
            "ai_recommendation": None,
        }

    closes = [float(r[1]) for r in price_rows]
    dates: list[date] = [r[0] for r in price_rows]
    latest_close = closes[0]
    latest_volume = price_rows[0][2]
    previous_close = closes[1] if len(closes) > 1 else None

    daily_change_pct = None
    if previous_close is not None and previous_close != 0:
        daily_change_pct = (latest_close - previous_close) / previous_close * 100

    return {
        "symbol": db_symbol,
        "company_name": company_name,
        "sector": sector,
        "industry": industry,
        "current_price": latest_close,
        "previous_close": previous_close,
        "daily_change_pct": daily_change_pct,
        "market_cap": float(market_cap) if market_cap is not None else None,
        "pe_ratio": float(pe_ratio) if pe_ratio is not None else None,
        "volume": latest_volume,
        "period_high": max(closes),
        "period_low": min(closes),
        "period_start_date": min(dates),
        "period_end_date": max(dates),
        "sentiment_score": sentiment_score,
        "ml_risk_score": ml_risk_score,
        "ai_recommendation": None,
    }


def get_top_movers(limit: int = 5) -> dict[str, list[dict[str, Any]]]:
    """
    Compute the top gaining and top losing companies by daily % change,
    for the Market Overview page's "Top Gainers & Top Losers" section.

    Args:
        limit: Maximum number of companies to return per list.

    Returns:
        A dict with "gainers" and "losers" keys, each a list of dicts with
        symbol, company_name, current_price, and daily_change_pct, sorted
        descending (gainers) / ascending (losers) by daily_change_pct.
        Companies with fewer than two days of price history are excluded,
        since a daily % change cannot be computed for them. Both lists are
        empty if no companies have at least two days of history yet.
    """
    query = """
        WITH ranked_prices AS (
            SELECT
                c.company_id,
                c.symbol,
                c.company_name,
                hp.close,
                ROW_NUMBER() OVER (PARTITION BY c.company_id ORDER BY hp.date DESC) AS rn
            FROM companies c
            JOIN historical_prices hp ON hp.company_id = c.company_id
        ),
        latest_two AS (
            SELECT
                company_id,
                symbol,
                company_name,
                MAX(CASE WHEN rn = 1 THEN close END) AS latest_close,
                MAX(CASE WHEN rn = 2 THEN close END) AS previous_close
            FROM ranked_prices
            WHERE rn IN (1, 2)
            GROUP BY company_id, symbol, company_name
        )
        SELECT
            symbol,
            company_name,
            latest_close,
            (latest_close - previous_close) / previous_close * 100 AS daily_change_pct
        FROM latest_two
        WHERE previous_close IS NOT NULL AND previous_close != 0
        ORDER BY daily_change_pct DESC;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()

    movers = [
        {
            "symbol": symbol,
            "company_name": company_name,
            "current_price": float(latest_close),
            "daily_change_pct": float(daily_change_pct),
        }
        for symbol, company_name, latest_close, daily_change_pct in rows
    ]

    return {
        "gainers": movers[:limit],
        "losers": list(reversed(movers[-limit:])) if movers else [],
    }


def get_sector_performance() -> list[dict[str, Any]]:
    """
    Compute average daily % change per sector, for the Market Overview
    page's "Sector Performance" section.

    Returns:
        A list of dicts with sector, company_count, and
        avg_daily_change_pct, sorted descending by avg_daily_change_pct.
        Sectors where no company has at least two days of price history
        are excluded. Returns an empty list if no companies are loaded yet.
    """
    query = """
        WITH ranked_prices AS (
            SELECT
                c.company_id,
                c.sector,
                hp.close,
                ROW_NUMBER() OVER (PARTITION BY c.company_id ORDER BY hp.date DESC) AS rn
            FROM companies c
            JOIN historical_prices hp ON hp.company_id = c.company_id
            WHERE c.sector IS NOT NULL
        ),
        latest_two AS (
            SELECT
                company_id,
                sector,
                MAX(CASE WHEN rn = 1 THEN close END) AS latest_close,
                MAX(CASE WHEN rn = 2 THEN close END) AS previous_close
            FROM ranked_prices
            WHERE rn IN (1, 2)
            GROUP BY company_id, sector
        )
        SELECT
            sector,
            COUNT(*) AS company_count,
            AVG((latest_close - previous_close) / previous_close * 100) AS avg_daily_change_pct
        FROM latest_two
        WHERE previous_close IS NOT NULL AND previous_close != 0
        GROUP BY sector
        ORDER BY avg_daily_change_pct DESC;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()

    return [
        {
            "sector": sector,
            "company_count": company_count,
            "avg_daily_change_pct": float(avg_daily_change_pct),
        }
        for sector, company_count, avg_daily_change_pct in rows
    ]
