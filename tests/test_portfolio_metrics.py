"""
test_portfolio_metrics.py

Unit tests for src/analytics/portfolio_metrics.py. No live Postgres is
required — src.utils.database.get_connection is mocked with a fake
cursor that returns scripted rows shaped exactly like the
watchlist_overview / company_sentiment_summary / latest_predictions
views defined in database/views.sql, per the blueprint's rule that every
phase's tests run against mocked DB boundaries when a live database isn't
available in the build environment.

Run with:
    python -m pytest tests/test_portfolio_metrics.py -v
"""

from contextlib import contextmanager
from datetime import date
from unittest.mock import MagicMock, patch

from src.analytics import portfolio_metrics


class FakeCursor:
    """Minimal cursor stand-in: returns pre-scripted rows for one execute() call."""

    def __init__(self, rows):
        self._rows = rows

    def execute(self, query, params=None):
        pass

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class FakeConnection:
    """Minimal connection stand-in whose cursor() yields a FakeCursor with the given rows."""

    def __init__(self, rows):
        self._rows = rows

    def cursor(self):
        return FakeCursor(self._rows)


def _fake_get_connection(rows):
    @contextmanager
    def _cm():
        yield FakeConnection(rows)
    return _cm


# One real position (AAPL, gain) + one real position (TSLA, loss) + one
# watch-only entry (MSFT, shares=0). Mirrors the exact column order
# get_portfolio_holdings() unpacks from its SELECT.
SAMPLE_ROWS = [
    (
        1, "AAPL", "Apple Inc.", "Technology",
        10.0, 150.00, date(2025, 1, 15),
        200.00, date(2026, 7, 8),
        2000.00, 1500.00, 500.00, 33.33,
        8, 2, 0, 0.25,
    ),
    (
        2, "TSLA", "Tesla, Inc.", "Consumer Cyclical",
        5.0, 300.00, date(2025, 3, 1),
        250.00, date(2026, 7, 8),
        1250.00, 1500.00, -250.00, -16.67,
        1, 5, 4, 0.72,
    ),
    (
        3, "MSFT", "Microsoft Corporation", "Technology",
        0.0, None, None,
        420.00, date(2026, 7, 8),
        None, None, None, None,
        None, None, None, None,
    ),
]


def test_get_portfolio_holdings_shapes_rows_correctly():
    with patch("src.analytics.portfolio_metrics.get_connection", _fake_get_connection(SAMPLE_ROWS)):
        holdings = portfolio_metrics.get_portfolio_holdings("jane")

    assert len(holdings) == 3

    aapl = next(h for h in holdings if h["symbol"] == "AAPL")
    assert aapl["is_position"] is True
    assert aapl["shares"] == 10.0
    assert aapl["market_value"] == 2000.00
    assert aapl["unrealized_pl"] == 500.00
    # positive=8, negative=2, neutral=0 -> (8-2)/10*100 = 60.0
    assert aapl["sentiment_score"] == 60.0
    assert aapl["ml_risk_score"] == 0.25

    msft = next(h for h in holdings if h["symbol"] == "MSFT")
    assert msft["is_position"] is False
    assert msft["shares"] == 0.0
    assert msft["market_value"] is None
    assert msft["sentiment_score"] is None


def test_get_portfolio_summary_aggregates_positions_only():
    with patch("src.analytics.portfolio_metrics.get_connection", _fake_get_connection(SAMPLE_ROWS)):
        summary = portfolio_metrics.get_portfolio_summary("jane")

    assert summary["position_count"] == 2
    assert summary["watch_only_count"] == 1
    assert summary["total_market_value"] == 2000.00 + 1250.00
    assert summary["total_cost_value"] == 1500.00 + 1500.00
    assert summary["total_unrealized_pl"] == (2000.00 + 1250.00) - (1500.00 + 1500.00)
    # avg sentiment across the two positions: (60.0 + (1-5)/10*100=-40.0) / 2 = 10.0
    assert summary["avg_sentiment_score"] == 10.0
    assert summary["avg_ml_risk_score"] == round((0.25 + 0.72) / 2, 4)


def test_get_portfolio_summary_handles_no_positions():
    watch_only_rows = [SAMPLE_ROWS[2]]  # MSFT only, shares = 0
    with patch("src.analytics.portfolio_metrics.get_connection", _fake_get_connection(watch_only_rows)):
        summary = portfolio_metrics.get_portfolio_summary("new_user")

    assert summary["position_count"] == 0
    assert summary["watch_only_count"] == 1
    assert summary["total_market_value"] == 0.0
    assert summary["total_unrealized_pl_pct"] is None
    assert summary["avg_sentiment_score"] is None


def test_get_sector_concentration_groups_and_computes_pct():
    with patch("src.analytics.portfolio_metrics.get_connection", _fake_get_connection(SAMPLE_ROWS)):
        breakdown = portfolio_metrics.get_sector_concentration("jane")

    assert len(breakdown) == 2  # Technology, Consumer Cyclical (MSFT excluded, no position)
    tech = next(b for b in breakdown if b["sector"] == "Technology")
    assert tech["market_value"] == 2000.00
    total = 2000.00 + 1250.00
    assert tech["pct_of_portfolio"] == round(2000.00 / total * 100, 2)
    # Sorted descending by market_value
    assert breakdown[0]["sector"] == "Technology"


def test_get_sector_concentration_empty_when_no_positions():
    watch_only_rows = [SAMPLE_ROWS[2]]
    with patch("src.analytics.portfolio_metrics.get_connection", _fake_get_connection(watch_only_rows)):
        breakdown = portfolio_metrics.get_sector_concentration("new_user")

    assert breakdown == []
