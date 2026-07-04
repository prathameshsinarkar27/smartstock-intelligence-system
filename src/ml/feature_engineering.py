"""
feature_engineering.py

Builds the feature matrix used by train_model.py, evaluate_model.py, and
predict.py: one row per (company, date), combining price-derived
technical indicators (reusing src/analytics/technical_indicators.py) with
a rolling sentiment signal (reusing src/sentiment/sentiment_pipeline.py's
output in sentiment_scores), plus a forward-looking trend label used only
for training.

"""

from datetime import date, timedelta
from typing import Any

import pandas as pd

from src.analytics.technical_indicators import compute_all_indicators
from src.utils.database import get_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Forward-looking label parameters (see module docstring).
FORWARD_HORIZON_DAYS = 5
UP_THRESHOLD = 0.02
DOWN_THRESHOLD = -0.02

# How many trailing calendar days of news sentiment feed the sentiment_7d
# feature. 7 days (rather than e.g. 1) smooths over the fact that news
# coverage is sparse and bursty for most companies.
SENTIMENT_LOOKBACK_DAYS = 7

# Minimum rows of price history a company needs before any feature row is
# considered usable (matches the longest lookback among the indicators
# below — SMA-50 — plus a small buffer so early-window edge effects don't
# leak into training).
MIN_PRICE_HISTORY_ROWS = 55

LABEL_TO_INT = {"down": 0, "flat": 1, "up": 2}
INT_TO_LABEL = {value: key for key, value in LABEL_TO_INT.items()}

# Single source of truth for column order — train_model.py, evaluate_model.py,
# and predict.py all import this rather than hardcoding column lists, so a
# model trained with one ordering is never fed a differently-ordered row.
FEATURE_COLUMNS = [
    "close_to_sma_20",
    "close_to_sma_50",
    "close_to_ema_12",
    "close_to_ema_26",
    "rsi_14",
    "macd_histogram_norm",
    "bollinger_percent_b",
    "bollinger_bandwidth",
    "return_1d",
    "return_5d",
    "volume_ratio_20d",
    "sentiment_7d",
]


def _get_tracked_companies(symbols: list[str] | None = None) -> list[tuple[int, str]]:
    """
    Fetch (company_id, symbol) pairs for companies to build features for.

    Args:
        symbols: If provided, restrict to these ticker symbols. If None,
            every company in the companies table is used.

    Returns:
        A list of (company_id, symbol) tuples, ordered by symbol.
    """
    upper_symbols = [s.upper() for s in symbols] if symbols else None

    query = """
        SELECT company_id, symbol
        FROM companies
        WHERE %(symbols)s::text[] IS NULL OR symbol = ANY(%(symbols)s)
        ORDER BY symbol;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, {"symbols": upper_symbols})
            rows = cur.fetchall()

    return [(company_id, symbol) for company_id, symbol in rows]


def _fetch_price_history(company_id: int) -> pd.DataFrame:
    """
    Fetch a company's full daily price/volume history, oldest first.

    Args:
        company_id: The company's surrogate key.

    Returns:
        A DataFrame with columns date, close, volume, sorted ascending by
        date. Empty DataFrame (same columns, zero rows) if no price
        history is loaded for this company.
    """
    query = """
        SELECT date, close, volume
        FROM historical_prices
        WHERE company_id = %s
        ORDER BY date ASC;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (company_id,))
            rows = cur.fetchall()

    return pd.DataFrame(rows, columns=["date", "close", "volume"])


def _fetch_sentiment_events(company_id: int) -> pd.DataFrame:
    """
    Fetch a company's scored news articles as (calendar date, signed
    sentiment value) pairs, for building the rolling sentiment_7d feature.

    Args:
        company_id: The company's surrogate key.

    Returns:
        A DataFrame with columns event_date (a date, not a timestamp —
        published_date's time-of-day doesn't matter here) and
        signed_value (+confidence_score for positive, -confidence_score
        for negative, 0.0 for neutral). Empty DataFrame (same columns,
        zero rows) if this company has no scored articles, or its
        articles have a null published_date.
    """
    query = """
        SELECT na.published_date, ss.sentiment, ss.confidence_score
        FROM news_articles na
        JOIN sentiment_scores ss ON ss.news_id = na.news_id
        WHERE na.company_id = %s
          AND na.published_date IS NOT NULL;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (company_id,))
            rows = cur.fetchall()

    events = []
    for published_date, sentiment, confidence_score in rows:
        confidence = float(confidence_score)
        if sentiment == "positive":
            signed_value = confidence
        elif sentiment == "negative":
            signed_value = -confidence
        else:
            signed_value = 0.0
        events.append({"event_date": published_date.date(), "signed_value": signed_value})

    return pd.DataFrame(events, columns=["event_date", "signed_value"])


def _rolling_sentiment_series(price_dates: pd.Series, sentiment_events: pd.DataFrame) -> pd.Series:
    """
    Compute a trailing-SENTIMENT_LOOKBACK_DAYS-day mean sentiment value
    aligned to each price date, with no lookahead (a price date only ever
    sees sentiment published on or before it).

    Args:
        price_dates: The date column of a company's price history,
            ascending.
        sentiment_events: Output of _fetch_sentiment_events() for the
            same company.

    Returns:
        A Series the same length as price_dates: the mean signed
        sentiment value across all scored articles published in the
        trailing SENTIMENT_LOOKBACK_DAYS calendar days (inclusive of the
        price date itself). 0.0 (a neutral prior, not a missing value)
        for any date with no scored articles in that window — including
        every date for a company with no scored news at all — so the
        feature is always numeric and never needs separate NaN handling
        downstream.
    """
    if sentiment_events.empty:
        return pd.Series(0.0, index=price_dates.index)

    # One row per calendar day with news, averaging same-day articles.
    daily = sentiment_events.groupby("event_date")["signed_value"].mean()
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()

    # Trailing window over calendar days (not trading days), so a
    # Saturday sentiment_7d value still reflects last week's news even
    # though the market was closed.
    rolling = daily.rolling(f"{SENTIMENT_LOOKBACK_DAYS}D", min_periods=1).mean()

    price_datetimes = pd.to_datetime(price_dates)
    # asof: for each price date, the most recent rolling value at or
    # before that date. Dates before any news exists get NaN, filled to
    # the same 0.0 neutral prior used for companies with no news at all.
    aligned = rolling.reindex(rolling.index.union(price_datetimes)).sort_index().ffill()
    result = aligned.reindex(price_datetimes).fillna(0.0)
    result.index = price_dates.index
    return result


def _build_labels(closes: pd.Series) -> pd.Series:
    """
    Build the forward-looking trend label for each date in a company's
    price history.

    Args:
        closes: Closing prices, ascending by date.

    Returns:
        A Series of "up"/"down"/"flat" strings, the same length as
        `closes`. The last FORWARD_HORIZON_DAYS entries are None — there
        isn't enough future price data yet to know their forward
        return — which is intentional: those rows are for inference
        (predict.py), not training.
    """
    forward_return = closes.shift(-FORWARD_HORIZON_DAYS) / closes - 1.0

    def _label(value: float) -> str | None:
        if pd.isna(value):
            return None
        if value > UP_THRESHOLD:
            return "up"
        if value < DOWN_THRESHOLD:
            return "down"
        return "flat"

    return forward_return.apply(_label)


def compute_features_for_company(company_id: int, symbol: str) -> pd.DataFrame:
    """
    Build the full feature (+ label) DataFrame for a single company.

    Args:
        company_id: The company's surrogate key.
        symbol: The company's ticker symbol (carried through as a column
            for convenience — callers combining multiple companies need
            it to tell rows apart).

    Returns:
        A DataFrame with columns: company_id, symbol, date, every column
        in FEATURE_COLUMNS, and label. Rows before MIN_PRICE_HISTORY_ROWS
        of history has accumulated have NaN feature values (insufficient
        indicator lookback); the last FORWARD_HORIZON_DAYS rows have a
        None label (see _build_labels). Empty DataFrame (same columns) if
        the company has fewer than MIN_PRICE_HISTORY_ROWS price rows
        loaded at all.
    """
    columns = ["company_id", "symbol", "date", *FEATURE_COLUMNS, "label"]

    prices = _fetch_price_history(company_id)
    if len(prices) < MIN_PRICE_HISTORY_ROWS:
        return pd.DataFrame(columns=columns)

    closes = prices["close"].astype(float)
    volumes = prices["volume"].astype(float)

    indicators = compute_all_indicators(closes.tolist())
    sma_20 = pd.Series(indicators["sma_20"], dtype="float64")
    sma_50 = pd.Series(indicators["sma_50"], dtype="float64")
    ema_12 = pd.Series(indicators["ema_12"], dtype="float64")
    ema_26 = pd.Series(indicators["ema_26"], dtype="float64")
    rsi_14 = pd.Series(indicators["rsi_14"], dtype="float64")
    macd_histogram = pd.Series(indicators["macd"]["histogram"], dtype="float64")
    bollinger_upper = pd.Series(indicators["bollinger"]["upper_band"], dtype="float64")
    bollinger_middle = pd.Series(indicators["bollinger"]["middle_band"], dtype="float64")
    bollinger_lower = pd.Series(indicators["bollinger"]["lower_band"], dtype="float64")

    sentiment_7d = _rolling_sentiment_series(prices["date"], _fetch_sentiment_events(company_id))

    features = pd.DataFrame({
        "company_id": company_id,
        "symbol": symbol,
        "date": prices["date"],
        "close_to_sma_20": closes / sma_20 - 1.0,
        "close_to_sma_50": closes / sma_50 - 1.0,
        "close_to_ema_12": closes / ema_12 - 1.0,
        "close_to_ema_26": closes / ema_26 - 1.0,
        "rsi_14": rsi_14,
        "macd_histogram_norm": macd_histogram / closes,
        "bollinger_percent_b": (closes - bollinger_lower) / (bollinger_upper - bollinger_lower),
        "bollinger_bandwidth": (bollinger_upper - bollinger_lower) / bollinger_middle,
        "return_1d": closes.pct_change(1),
        "return_5d": closes.pct_change(5),
        "volume_ratio_20d": volumes / volumes.rolling(20).mean(),
        "sentiment_7d": sentiment_7d,
        "label": _build_labels(closes),
    })

    return features[columns]


def build_feature_dataset(symbols: list[str] | None = None) -> pd.DataFrame:
    """
    Build the combined feature (+ label) dataset across all tracked
    companies (or a subset).

    Args:
        symbols: If provided, restrict to these ticker symbols. If None,
            every company in the companies table is used.

    Returns:
        A DataFrame combining compute_features_for_company()'s output for
        every matching company, with a fresh 0-based index. Companies
        with fewer than MIN_PRICE_HISTORY_ROWS price rows contribute no
        rows at all. Empty DataFrame (with the expected columns) if no
        companies match or none have enough history yet.
    """
    companies = _get_tracked_companies(symbols)

    if not companies:
        logger.warning("build_feature_dataset: no matching companies found.")
        return pd.DataFrame(columns=["company_id", "symbol", "date", *FEATURE_COLUMNS, "label"])

    frames = []
    for company_id, symbol in companies:
        company_features = compute_features_for_company(company_id, symbol)
        if company_features.empty:
            logger.info(
                "Skipping %s: fewer than %d price rows loaded.", symbol, MIN_PRICE_HISTORY_ROWS
            )
            continue
        frames.append(company_features)

    if not frames:
        return pd.DataFrame(columns=["company_id", "symbol", "date", *FEATURE_COLUMNS, "label"])

    return pd.concat(frames, ignore_index=True)


def build_training_rows(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Filter a feature dataset down to rows usable for training/evaluation:
    complete features and a non-null label.

    Args:
        dataset: Output of build_feature_dataset().

    Returns:
        The subset of rows with no NaN in FEATURE_COLUMNS and a non-null
        label, with a fresh 0-based index.
    """
    usable = dataset.dropna(subset=[*FEATURE_COLUMNS, "label"])
    return usable.reset_index(drop=True)


def build_latest_inference_rows(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce a feature dataset to one row per company: its most recent date
    with complete features, regardless of whether it has a label (rows
    usable for inference are, by construction, exactly the ones near the
    tail that lack a label — see build_feature_dataset's docstring).

    Args:
        dataset: Output of build_feature_dataset().

    Returns:
        One row per symbol (its latest complete-feature row), with a
        fresh 0-based index. A company contributes no row if none of its
        rows have complete features (shouldn't happen for any company
        that cleared MIN_PRICE_HISTORY_ROWS, barring a data gap).
    """
    complete = dataset.dropna(subset=FEATURE_COLUMNS)
    if complete.empty:
        return complete.reset_index(drop=True)

    latest = complete.sort_values("date").groupby("symbol", as_index=False).tail(1)
    return latest.reset_index(drop=True)
