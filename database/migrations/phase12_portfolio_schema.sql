-- phase12_portfolio_schema.sql
--
-- SmartStock Intelligence Platform — Phase 12 Migration
-- Portfolio Analyzer: extend `watchlist` into a true portfolio table.
--
-- Context (see docs/PHASE_12_SETUP_GUIDE.md and blueprint section 12/14):
-- Through Phase 11 the `watchlist` table only recorded which symbols a
-- user follows (user_name + symbol), with no share quantity or cost
-- basis, so no real profit/loss could be computed. Phase 12 was
-- confirmed by the user to mean a TRUE portfolio: this migration adds
-- `shares` and `avg_cost_basis` (plus an optional `purchased_at`) so
-- unrealized P&L can be computed per holding.
--
-- Backward compatible: existing rows get shares = 0, avg_cost_basis =
-- NULL, i.e. they remain plain "watch this symbol, own none of it"
-- entries — no data is lost or reinterpreted. A row only becomes a real
-- holding once shares > 0 and avg_cost_basis is set, via the Phase 12
-- "Add / Update Holding" form (src/api/routes/portfolio.py).
--
-- Run this AFTER tables.sql / views.sql have been applied once already:
--   psql -U postgres -d smartstock -f database/migrations/phase12_portfolio_schema.sql
--
-- This migration is also folded into database/tables.sql and
-- database/views.sql directly, so a fresh `psql -f database/tables.sql`
-- on a brand-new database already includes these columns — this file
-- exists so an existing Phase 0-11 database can be upgraded in place
-- without dropping data.

-- ----------------------------------------------------------------------
-- Step 1 — Add portfolio columns to watchlist
-- ----------------------------------------------------------------------
ALTER TABLE watchlist
    ADD COLUMN IF NOT EXISTS shares          NUMERIC(18, 6)  NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS avg_cost_basis  NUMERIC(14, 4),
    ADD COLUMN IF NOT EXISTS purchased_at    DATE;

-- A holding must either be "watch only" (shares = 0, no cost basis) or
-- a real position (shares > 0 with a recorded average cost basis) — it
-- cannot be a half-state like shares > 0 with no cost basis, which
-- would make P&L undefined.
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
-- Superseded by the version in database/views.sql, repeated here so this
-- migration is self-contained and idempotent. Columns are ordered with
-- shares/avg_cost_basis/purchased_at AFTER the pre-existing columns
-- (watchlist_id..latest_price_date) rather than inserted in the middle:
-- CREATE OR REPLACE VIEW can only append new trailing columns to an
-- existing view, not reorder or insert them, or Postgres raises
-- "cannot change name of view column ... to ...". DROP + CREATE is used
-- instead of CREATE OR REPLACE for extra safety, since this migration
-- may be re-run against a database where a previous run partially
-- applied a different column order.
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
