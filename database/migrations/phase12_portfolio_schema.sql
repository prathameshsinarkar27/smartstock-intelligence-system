-- phase12_portfolio_schema.sql
-- SmartStock Intelligence Platform — Portfolio Migration
-- Extends watchlist with portfolio holding details.

-- Adds shares, cost basis, and purchase date so P&L can be calculated.
-- Existing rows remain watch-only entries with shares = 0.

-- Run after tables.sql and views.sql.
-- Example:
-- psql -U postgres -d smartstock -f database/migrations/phase12_portfolio_schema.sql

-- ----------------------------------------------------------------------
-- Step 1 — Add portfolio columns to watchlist
-- ----------------------------------------------------------------------
ALTER TABLE watchlist
    ADD COLUMN IF NOT EXISTS shares          NUMERIC(18, 6)  NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS avg_cost_basis  NUMERIC(14, 4),
    ADD COLUMN IF NOT EXISTS purchased_at    DATE;

-- Require either a watch-only entry or a complete position.
ALTER TABLE watchlist
    DROP CONSTRAINT IF EXISTS chk_watchlist_shares_cost_consistency;

ALTER TABLE watchlist
    ADD CONSTRAINT chk_watchlist_shares_cost_consistency
    CHECK (
        (shares = 0 AND avg_cost_basis IS NULL)
        OR (shares > 0 AND avg_cost_basis IS NOT NULL)
    );

ALTER TABLE watchlist
    DROP CONSTRAINT IF EXISTS chk_watchlist_shares_nonnegative;

ALTER TABLE watchlist
    ADD CONSTRAINT chk_watchlist_shares_nonnegative CHECK (shares >= 0);

ALTER TABLE watchlist
    DROP CONSTRAINT IF EXISTS chk_watchlist_cost_basis_positive;

ALTER TABLE watchlist
    ADD CONSTRAINT chk_watchlist_cost_basis_positive
    CHECK (avg_cost_basis IS NULL OR avg_cost_basis > 0);

COMMENT ON COLUMN watchlist.shares IS
    'Shares held for this symbol. 0 means "watching only, no position" (Phase 0-11 behavior preserved).';
COMMENT ON COLUMN watchlist.avg_cost_basis IS
    'Average purchase price per share. NULL when shares = 0.';
COMMENT ON COLUMN watchlist.purchased_at IS
    'Optional date the position was opened / last updated. Informational only, not used in P&L math.';

-- ----------------------------------------------------------------------
-- Step 2 — Recreate watchlist_overview with portfolio math
-- ----------------------------------------------------------------------
-- Recreates the view with holding details and P&L calculations.
-- DROP + CREATE ensures the column order can be changed safely.
DROP VIEW IF EXISTS watchlist_overview;

CREATE VIEW watchlist_overview AS
SELECT
    w.watchlist_id,
    w.user_name,
    w.symbol,
    c.company_id,
    c.company_name,
    c.sector,
    lp.close AS latest_close,
    lp.date  AS latest_price_date,
    w.shares,
    w.avg_cost_basis,
    w.purchased_at,
    (w.shares * lp.close)              AS market_value,
    (w.shares * w.avg_cost_basis)      AS cost_value,
    (w.shares * lp.close) - (w.shares * w.avg_cost_basis) AS unrealized_pl,
    CASE
        WHEN w.shares > 0 AND w.avg_cost_basis IS NOT NULL AND w.avg_cost_basis != 0
            THEN ROUND((((lp.close - w.avg_cost_basis) / w.avg_cost_basis) * 100)::numeric, 2)
        ELSE NULL
    END AS unrealized_pl_pct
FROM watchlist w
LEFT JOIN companies c ON c.symbol = w.symbol
LEFT JOIN latest_prices lp ON lp.company_id = c.company_id;

COMMENT ON VIEW watchlist_overview IS
    'Per-user watchlist/portfolio joined with company info, latest price, and unrealized P&L (Phase 12).';
