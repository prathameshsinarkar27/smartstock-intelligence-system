-- tables.sql
--
-- SmartStock Intelligence Platform — Table Definitions
--
-- Defines the six core tables:
--   companies, historical_prices, news_articles, sentiment_scores,
--   predictions, watchlist
--
-- Run this AFTER schema.sql, connected to the 'smartstock' database:
--   psql -U postgres -d smartstock -f database/tables.sql


-- ----------------------------------------------------------------------
-- Table: companies
-- ----------------------------------------------------------------------
-- One row per tracked stock symbol. Populated by src/etl/load_to_db.py
-- from src/ingestion/fetch_company_data.py output (Phase 3).
CREATE TABLE IF NOT EXISTS companies (
    company_id      SERIAL PRIMARY KEY,
    symbol          VARCHAR(20)     NOT NULL UNIQUE,
    company_name    VARCHAR(255),
    sector          VARCHAR(100),
    industry        VARCHAR(100),
    market_cap      NUMERIC(20, 2),
    pe_ratio        NUMERIC(10, 4),
    eps             NUMERIC(10, 4),
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT now()
);

COMMENT ON TABLE companies IS 'One row per tracked stock symbol; company fundamentals.';

-- ----------------------------------------------------------------------
-- Table: historical_prices
-- ----------------------------------------------------------------------
-- One row per (company, date) daily OHLCV candle. Populated by
-- src/etl/load_to_db.py from src/ingestion/fetch_stock_data.py output.
CREATE TABLE IF NOT EXISTS historical_prices (
    price_id        BIGSERIAL PRIMARY KEY,
    company_id      INTEGER         NOT NULL REFERENCES companies(company_id) ON DELETE CASCADE,
    date            DATE            NOT NULL,
    open            NUMERIC(14, 4)  NOT NULL,
    high            NUMERIC(14, 4)  NOT NULL,
    low             NUMERIC(14, 4)  NOT NULL,
    close           NUMERIC(14, 4)  NOT NULL,
    volume          BIGINT          NOT NULL,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),

    -- A company should only have one candle per calendar date.
    CONSTRAINT uq_historical_prices_company_date UNIQUE (company_id, date),

    -- Sanity constraints: high must be the max and low must be the min
    -- of the day's price action.
    CONSTRAINT chk_price_high_low CHECK (high >= low),
    CONSTRAINT chk_price_nonnegative CHECK (open >= 0 AND high >= 0 AND low >= 0 AND close >= 0),
    CONSTRAINT chk_volume_nonnegative CHECK (volume >= 0)
);

COMMENT ON TABLE historical_prices IS 'Daily OHLCV price candles per company.';

CREATE INDEX IF NOT EXISTS idx_historical_prices_company_date
    ON historical_prices (company_id, date DESC);

-- ----------------------------------------------------------------------
-- Table: news_articles
-- ----------------------------------------------------------------------
-- One row per news article fetched per company. Populated by
-- src/etl/load_to_db.py from src/ingestion/fetch_news.py output.
CREATE TABLE IF NOT EXISTS news_articles (
    news_id         BIGSERIAL PRIMARY KEY,
    company_id      INTEGER         NOT NULL REFERENCES companies(company_id) ON DELETE CASCADE,
    title           TEXT            NOT NULL,
    content         TEXT,
    source          VARCHAR(255),
    published_date  TIMESTAMPTZ,
    url             TEXT,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),

    -- Prevent loading the exact same article twice for the same company.
    CONSTRAINT uq_news_articles_company_url UNIQUE (company_id, url)
);

COMMENT ON TABLE news_articles IS 'News articles fetched per company, prior to sentiment scoring.';

CREATE INDEX IF NOT EXISTS idx_news_articles_company_date
    ON news_articles (company_id, published_date DESC);

-- ----------------------------------------------------------------------
-- Table: sentiment_scores
-- ----------------------------------------------------------------------
-- One row per news article's sentiment result. Populated by
-- src/sentiment/sentiment_pipeline.py (Phase 7).
CREATE TABLE IF NOT EXISTS sentiment_scores (
    score_id            BIGSERIAL PRIMARY KEY,
    news_id             BIGINT          NOT NULL REFERENCES news_articles(news_id) ON DELETE CASCADE,
    sentiment           VARCHAR(20)     NOT NULL,
    confidence_score    NUMERIC(5, 4)   NOT NULL,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),

    -- One sentiment result per article (re-scoring should UPDATE, not insert).
    CONSTRAINT uq_sentiment_scores_news_id UNIQUE (news_id),

    CONSTRAINT chk_sentiment_value
        CHECK (sentiment IN ('positive', 'negative', 'neutral')),
    CONSTRAINT chk_confidence_range
        CHECK (confidence_score >= 0 AND confidence_score <= 1)
);

COMMENT ON TABLE sentiment_scores IS 'Sentiment classification result per news article.';

CREATE INDEX IF NOT EXISTS idx_sentiment_scores_news_id
    ON sentiment_scores (news_id);

-- ----------------------------------------------------------------------
-- Table: predictions
-- ----------------------------------------------------------------------
-- One row per (company, prediction_date) ML prediction. Populated by
-- src/ml/predict.py (Phase 8).
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id       BIGSERIAL PRIMARY KEY,
    company_id           INTEGER         NOT NULL REFERENCES companies(company_id) ON DELETE CASCADE,
    prediction_date       DATE            NOT NULL,
    trend_prediction      VARCHAR(20)     NOT NULL,
    risk_score            NUMERIC(5, 4)   NOT NULL,
    created_at            TIMESTAMPTZ     NOT NULL DEFAULT now(),

    -- One prediction per company per date (re-running inference updates it).
    CONSTRAINT uq_predictions_company_date UNIQUE (company_id, prediction_date),

    CONSTRAINT chk_trend_prediction_value
        CHECK (trend_prediction IN ('up', 'down', 'flat')),
    CONSTRAINT chk_risk_score_range
        CHECK (risk_score >= 0 AND risk_score <= 1)
);

COMMENT ON TABLE predictions IS 'ML-generated trend and risk predictions per company per date.';

CREATE INDEX IF NOT EXISTS idx_predictions_company_date
    ON predictions (company_id, prediction_date DESC);

-- ----------------------------------------------------------------------
-- Table: watchlist
-- ----------------------------------------------------------------------
-- User-curated list of symbols, used by the Portfolio Analyzer (Phase 12).
-- user_name is a plain string rather than a foreign key to a users table,
-- since the blueprint does not define user authentication/accounts.
--
-- Phase 12 note: shares/avg_cost_basis/purchased_at were added so this
-- table can represent a TRUE portfolio (real P&L), not just a symbol
-- watchlist. A row with shares = 0 (the default) is still a plain
-- "watching only" entry exactly like Phases 0-11 — it only becomes a
-- real position once shares > 0 and avg_cost_basis is set. See
-- database/migrations/phase12_portfolio_schema.sql for the in-place
-- upgrade path on an existing database.
CREATE TABLE IF NOT EXISTS watchlist (
    watchlist_id    BIGSERIAL PRIMARY KEY,
    user_name       VARCHAR(100)    NOT NULL,
    symbol          VARCHAR(20)     NOT NULL,
    shares          NUMERIC(18, 6)  NOT NULL DEFAULT 0,
    avg_cost_basis  NUMERIC(14, 4),
    purchased_at    DATE,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),

    -- A user should not be able to add the same symbol twice.
    CONSTRAINT uq_watchlist_user_symbol UNIQUE (user_name, symbol),

    -- Either "watch only" (no position) or a fully-specified position —
    -- never a half-state where P&L would be undefined.
    CONSTRAINT chk_watchlist_shares_cost_consistency
        CHECK (
            (shares = 0 AND avg_cost_basis IS NULL)
            OR (shares > 0 AND avg_cost_basis IS NOT NULL)
        ),
    CONSTRAINT chk_watchlist_shares_nonnegative CHECK (shares >= 0),
    CONSTRAINT chk_watchlist_cost_basis_positive
        CHECK (avg_cost_basis IS NULL OR avg_cost_basis > 0)
);

COMMENT ON TABLE watchlist IS 'User-curated stock watchlist / portfolio (Phase 12 adds real positions: shares + avg_cost_basis).';
COMMENT ON COLUMN watchlist.shares IS 'Shares held for this symbol. 0 means "watching only, no position".';
COMMENT ON COLUMN watchlist.avg_cost_basis IS 'Average purchase price per share. NULL when shares = 0.';
COMMENT ON COLUMN watchlist.purchased_at IS 'Optional date the position was opened / last updated. Informational only.';

CREATE INDEX IF NOT EXISTS idx_watchlist_user_name
    ON watchlist (user_name);
