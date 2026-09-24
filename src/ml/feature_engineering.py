"""
feature_engineering.py

Builds the feature matrix used by the ML pipeline by combining technical
indicators, rolling sentiment, and forward-looking training labels.
"""

from datetime import date, timedelta
from typing import Any

import pandas as pd

from src.analytics.technical_indicators import compute_all_indicators
from src.utils.database import get_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Forward-looking label parameters.
FORWARD_HORIZON_DAYS = 5
UP_THRESHOLD = 0.02
DOWN_THRESHOLD = -0.02

# Trailing calendar days used for the sentiment feature.
SENTIMENT_LOOKBACK_DAYS = 7

# Minimum history required for reliable feature calculation.
MIN_PRICE_HISTORY_ROWS = 55

LABEL_TO_INT = {"down": 0, "flat": 1, "up": 2}
INT_TO_LABEL = {value: key for key, value in LABEL_TO_INT.items()}

# Shared feature order used by training, evaluation, and inference.
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
    Fetch company IDs and symbols used for feature generation.

    Args:
        symbols: Optional list of ticker symbols to filter by.

    Returns:
        Company ID and symbol pairs ordered by symbol.
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
    Fetch a company's daily price and volume history.

    Args:
        company_id: Company's database ID.

    Returns:
        DataFrame with date, close, and volume sorted by date.
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
    Fetch scored news sentiment events for a company.

    Args:
        company_id: Company's database ID.

    Returns:
        DataFrame with event dates and signed sentiment values.
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
    Compute trailing sentiment aligned to each price date without lookahead.

    Args:
        price_dates: Ascending price dates.
        sentiment_events: Scored sentiment events for the company.

    Returns:
        Rolling mean sentiment for each price date, using 0.0 when
        no scored articles are available.
    """
    if sentiment_events.empty:
        return pd.Series(0.0, index=price_dates.index)

    # Average multiple articles published on the same calendar day.
    daily = sentiment_events.groupby("event_date")["signed_value"].mean()
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()

    # Use calendar days so weekend news remains part of the rolling window.
    rolling = daily.rolling(f"{SENTIMENT_LOOKBACK_DAYS}D", min_periods=1).mean()

    price_datetimes = pd.to_datetime(price_dates)
    # Align each price date with the latest available sentiment value.
    aligned = rolling.reindex(rolling.index.union(price_datetimes)).sort_index().ffill()
    result = aligned.reindex(price_datetimes).fillna(0.0)
    result.index = price_dates.index
    return result


def _build_labels(closes: pd.Series) -> pd.Series:
    """
    Build forward-looking trend labels from closing prices.

    Args:
        closes: Closing prices sorted by date.

    Returns:
        Series containing "up", "down", or "flat" labels.
        Final rows without future data receive None.
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
    Build features and labels for a single company.

    Args:
        company_id: Company's database ID.
        symbol: Stock ticker symbol.

    Returns:
        DataFrame containing company metadata, features, and labels.
        Returns an empty DataFrame when insufficient price history exists.
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
    Build the combined feature dataset across tracked companies.

    Args:
        symbols: Optional list of ticker symbols to include.

    Returns:
        Combined feature DataFrame with a fresh index.
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
    Filter a feature dataset to rows with complete features and labels.

    Args:
        dataset: Feature dataset.

    Returns:
        Training-ready rows with a fresh index.
    """
    usable = dataset.dropna(subset=[*FEATURE_COLUMNS, "label"])
    return usable.reset_index(drop=True)


def build_latest_inference_rows(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Select the latest complete-feature row for each company.

    Args:
        dataset: Feature dataset.

    Returns:
        One latest complete-feature row per symbol.
    """
    complete = dataset.dropna(subset=FEATURE_COLUMNS)
    if complete.empty:
        return complete.reset_index(drop=True)

    latest = complete.sort_values("date").groupby("symbol", as_index=False).tail(1)
    return latest.reset_index(drop=True)
