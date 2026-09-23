# kpi_calculator.py
#
# Computes KPI metrics for the Market Overview and Company Detail pages:
# price, daily change, market cap, volume, P/E, and available high/low range.


from datetime import date
from typing import Any

from src.utils.database import get_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_market_overview_kpis() -> dict[str, Any]:
    """
    Compute market-wide KPI metrics for the Market Overview page.

    Returns:
        Total companies and sectors, advancing/declining counts,
        and average daily percentage change.
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
    Compute the signed sentiment score for a company.

    Returns:
        Score from -100 to 100, or None if no scored articles exist.
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
    Fetch the latest ML risk score for a company.

    Returns:
        Risk score from 0 to 1, or None if no prediction exists.
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
    Compute KPI metrics for a company's detail page.

    High/low values use the available price history rather than assuming
    a full 52-week period.

    Returns:
        Company details, price metrics, sentiment, ML risk score,
        and available price range.
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

        # Calculate sentiment and ML risk while the connection is open.
        sentiment_score = _get_company_sentiment_score(conn, company_id)
        ml_risk_score = _get_company_ml_risk_score(conn, company_id)

    if not price_rows:
        # Company exists but has no price history.
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
        # Reserved for future AI outlook integration.
        "ai_recommendation": None,
    }


def get_top_movers(limit: int = 5) -> dict[str, list[dict[str, Any]]]:
    """
    Compute the top gaining and losing companies by daily percentage change.

    Companies with fewer than two price records are excluded.
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
    Compute average daily percentage change for each sector.

    Sectors without sufficient price history are excluded.
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
