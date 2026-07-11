-- views.sql
--
-- SmartStock Intelligence Platform — Reporting Views

-- Reporting views used by:
-- - Analytics
-- - Streamlit Dashboard
-- - FastAPI
-- - Portfolio Analyzer

-- Run this AFTER tables.sql:
--   psql -U postgres -d smartstock -f database/views.sql

-- ----------------------------------------------------------------------
-- View: latest_prices
-- ----------------------------------------------------------------------
-- The most recent price candle per company, joined with company info.
-- Used by KPI cards (Phase 5) to show current price at a glance.
CREATE OR REPLACE VIEW latest_prices AS
SELECT
    c.company_id,
    c.symbol,
    c.company_name,
    hp.date,
    hp.open,
    hp.high,
    hp.low,
    hp.close,
    hp.volume
FROM companies c
JOIN historical_prices hp ON hp.company_id = c.company_id
WHERE hp.date = (
    SELECT MAX(hp2.date)
    FROM historical_prices hp2
    WHERE hp2.company_id = c.company_id
);

COMMENT ON VIEW latest_prices IS 'Most recent OHLCV candle per company.';

-- ----------------------------------------------------------------------
-- View: company_sentiment_summary
-- ----------------------------------------------------------------------
-- Aggregated sentiment counts and average confidence per company, across
-- all scored news articles. Used by the Sentiment Analysis dashboard page
-- (Phase 7).
CREATE OR REPLACE VIEW company_sentiment_summary AS
SELECT
    c.company_id,
    c.symbol,
    c.company_name,
    COUNT(*) FILTER (WHERE ss.sentiment = 'positive') AS positive_count,
    COUNT(*) FILTER (WHERE ss.sentiment = 'negative') AS negative_count,
    COUNT(*) FILTER (WHERE ss.sentiment = 'neutral')  AS neutral_count,
    COUNT(*)                                          AS total_scored_articles,
    ROUND(AVG(ss.confidence_score), 4)                AS avg_confidence_score
FROM companies c
JOIN news_articles na ON na.company_id = c.company_id
JOIN sentiment_scores ss ON ss.news_id = na.news_id
GROUP BY c.company_id, c.symbol, c.company_name;

COMMENT ON VIEW company_sentiment_summary IS 'Aggregated sentiment distribution per company.';

-- ----------------------------------------------------------------------
-- View: latest_predictions
-- ----------------------------------------------------------------------
-- The most recent ML prediction per company, joined with company info.
-- Used by the ML Predictions dashboard page (Phase 8) and the GenAI
-- assistant (Phase 10) when explaining current risk/trend outlook.
CREATE OR REPLACE VIEW latest_predictions AS
SELECT
    c.company_id,
    c.symbol,
    c.company_name,
    p.prediction_date,
    p.trend_prediction,
    p.risk_score
FROM companies c
JOIN predictions p ON p.company_id = c.company_id
WHERE p.prediction_date = (
    SELECT MAX(p2.prediction_date)
    FROM predictions p2
    WHERE p2.company_id = c.company_id
);

COMMENT ON VIEW latest_predictions IS 'Most recent trend/risk prediction per company.';

-- ----------------------------------------------------------------------
-- View: watchlist_overview
-- ----------------------------------------------------------------------
-- Joins each user's watchlist entries with current company info and
-- latest price, for the Portfolio Analyzer (Phase 12).
-- Phase 12 adds shares/avg_cost_basis/purchased_at plus computed
-- market_value / cost_value / unrealized_pl / unrealized_pl_pct, so the
-- Portfolio Analyzer page and src/analytics/portfolio_metrics.py can read
-- P&L straight from this view instead of recomputing it in Python.
CREATE OR REPLACE VIEW watchlist_overview AS
SELECT
    w.watchlist_id,
    w.user_name,
    w.symbol,
    c.company_id,
    c.company_name,
    c.sector,
    lp.close AS latest_close,
    lp.date AS latest_price_date,
    w.shares,
    w.avg_cost_basis,
    w.purchased_at,
    (w.shares * lp.close) AS market_value,
    (w.shares * w.avg_cost_basis) AS cost_value,
    (w.shares * lp.close) - (w.shares * w.avg_cost_basis) AS unrealized_pl,
    CASE
        WHEN w.shares > 0 AND w.avg_cost_basis IS NOT NULL AND w.avg_cost_basis != 0
            THEN ROUND((((lp.close - w.avg_cost_basis) / w.avg_cost_basis) * 100)::numeric, 2)
        ELSE NULL
    END AS unrealized_pl_pct
FROM watchlist w
LEFT JOIN companies c ON c.symbol = w.symbol
LEFT JOIN latest_prices lp ON lp.company_id = c.company_id;

COMMENT ON VIEW watchlist_overview IS 'Per-user watchlist/portfolio joined with company info, latest price, and unrealized P&L (Phase 12).';
