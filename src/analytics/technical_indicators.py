"""
technical_indicators.py

Computes technical indicators including SMA, EMA, RSI, MACD,
and Bollinger Bands from historical price data.
"""

from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_SMA_PERIODS = (20, 50)
DEFAULT_EMA_PERIODS = (12, 26)
DEFAULT_RSI_PERIOD = 14
DEFAULT_MACD_FAST = 12
DEFAULT_MACD_SLOW = 26
DEFAULT_MACD_SIGNAL = 9
DEFAULT_BOLLINGER_PERIOD = 20
DEFAULT_BOLLINGER_STD_DEV = 2.0


def simple_moving_average(closes: list[float], period: int) -> list[float | None]:
    """
    Compute the Simple Moving Average (SMA).

    Args:
        closes: Closing prices, oldest first.
        period: Rolling window size.

    Returns:
        SMA values with None where insufficient data exists.
    """
    result: list[float | None] = [None] * len(closes)

    if period <= 0 or len(closes) < period:
        return result

    window_sum = sum(closes[:period])
    result[period - 1] = window_sum / period

    for i in range(period, len(closes)):
        window_sum += closes[i] - closes[i - period]
        result[i] = window_sum / period

    return result


def exponential_moving_average(closes: list[float], period: int) -> list[float | None]:
    """
    Compute the Exponential Moving Average (EMA).

    Args:
        closes: Closing prices, oldest first.
        period: Smoothing window size.

    Returns:
        EMA values with None where insufficient data exists.
    """
    result: list[float | None] = [None] * len(closes)

    if period <= 0 or len(closes) < period:
        return result

    smoothing_factor = 2 / (period + 1)

    # Seed with a simple average of the first `period` closes.
    seed = sum(closes[:period]) / period
    result[period - 1] = seed

    previous_ema = seed
    for i in range(period, len(closes)):
        current_ema = closes[i] * smoothing_factor + previous_ema * (1 - smoothing_factor)
        result[i] = current_ema
        previous_ema = current_ema

    return result


def relative_strength_index(closes: list[float], period: int = DEFAULT_RSI_PERIOD) -> list[float | None]:
    """
    Compute the Relative Strength Index (RSI) using Wilder's smoothing.

    Args:
        closes: Closing prices, oldest first.
        period: Lookback window.

    Returns:
        RSI values from 0-100 with None where insufficient data exists.
    """
    result: list[float | None] = [None] * len(closes)

    if period <= 0 or len(closes) < period + 1:
        return result

    changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(c, 0) for c in changes]
    losses = [max(-c, 0) for c in changes]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    def _rsi_from_averages(avg_gain: float, avg_loss: float) -> float:
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    # First RSI value aligns with closes[period].
    result[period] = _rsi_from_averages(avg_gain, avg_loss)

    for i in range(period, len(changes)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        # changes[i] is the change ending at closes[i + 1].
        result[i + 1] = _rsi_from_averages(avg_gain, avg_loss)

    return result


def macd(
    closes: list[float],
    fast_period: int = DEFAULT_MACD_FAST,
    slow_period: int = DEFAULT_MACD_SLOW,
    signal_period: int = DEFAULT_MACD_SIGNAL,
) -> dict[str, list[float | None]]:
    """
    Compute MACD, signal line, and histogram.

    Args:
        closes: Closing prices, oldest first.
        fast_period: Fast EMA window.
        slow_period: Slow EMA window.
        signal_period: Signal EMA window.

    Returns:
        MACD line, signal line, and histogram values.
    """
    fast_ema = exponential_moving_average(closes, fast_period)
    slow_ema = exponential_moving_average(closes, slow_period)

    macd_line: list[float | None] = [
        (f - s) if f is not None and s is not None else None
        for f, s in zip(fast_ema, slow_ema)
    ]

    # Calculate the signal EMA from the valid MACD values.
    first_valid_index = next((i for i, v in enumerate(macd_line) if v is not None), None)

    signal_line: list[float | None] = [None] * len(closes)
    histogram: list[float | None] = [None] * len(closes)

    if first_valid_index is not None:
        macd_tail = [v for v in macd_line[first_valid_index:]]
        signal_tail = exponential_moving_average(macd_tail, signal_period)

        for offset, value in enumerate(signal_tail):
            signal_line[first_valid_index + offset] = value

        for i in range(len(closes)):
            if macd_line[i] is not None and signal_line[i] is not None:
                histogram[i] = macd_line[i] - signal_line[i]

    return {
        "macd_line": macd_line,
        "signal_line": signal_line,
        "histogram": histogram,
    }


def bollinger_bands(
    closes: list[float],
    period: int = DEFAULT_BOLLINGER_PERIOD,
    num_std_dev: float = DEFAULT_BOLLINGER_STD_DEV,
) -> dict[str, list[float | None]]:
    """
    Compute Bollinger Bands from a rolling mean and standard deviation.

    Args:
        closes: Closing prices, oldest first.
        period: Rolling window size.
        num_std_dev: Standard deviation multiplier.

    Returns:
        Middle, upper, and lower Bollinger Bands.
    """
    middle_band = simple_moving_average(closes, period)
    upper_band: list[float | None] = [None] * len(closes)
    lower_band: list[float | None] = [None] * len(closes)

    if period <= 0 or len(closes) < period:
        return {"middle_band": middle_band, "upper_band": upper_band, "lower_band": lower_band}

    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1: i + 1]
        mean = middle_band[i]
        variance = sum((x - mean) ** 2 for x in window) / period
        std_dev = variance ** 0.5

        upper_band[i] = mean + num_std_dev * std_dev
        lower_band[i] = mean - num_std_dev * std_dev

    return {"middle_band": middle_band, "upper_band": upper_band, "lower_band": lower_band}


def interpret_rsi(latest_rsi: float | None) -> str:
    """
    Convert the latest RSI value into a signal label.

    Args:
        latest_rsi: Most recent RSI value.

    Returns:
        Overbought, Oversold, Neutral, or —.
    """
    if latest_rsi is None:
        return "—"
    if latest_rsi > 70:
        return "Overbought"
    if latest_rsi < 30:
        return "Oversold"
    return "Neutral"


def interpret_macd_crossover(latest_macd_line: float | None, latest_signal_line: float | None) -> str:
    """
    Convert the MACD/signal relationship into a signal label.

    Args:
        latest_macd_line: Latest MACD line value.
        latest_signal_line: Latest signal line value.

    Returns:
        Bullish, Bearish, or — when data is unavailable.
    """
    if latest_macd_line is None or latest_signal_line is None:
        return "—"
    return "Bullish" if latest_macd_line > latest_signal_line else "Bearish"


def compute_all_indicators(closes: list[float]) -> dict[str, object]:
    """
    Compute all technical indicators using default periods.

    Args:
        closes: Closing prices, oldest first.

    Returns:
        Dictionary containing SMA, EMA, RSI, MACD, and Bollinger Bands.
    """
    return {
        "sma_20": simple_moving_average(closes, DEFAULT_SMA_PERIODS[0]),
        "sma_50": simple_moving_average(closes, DEFAULT_SMA_PERIODS[1]),
        "ema_12": exponential_moving_average(closes, DEFAULT_EMA_PERIODS[0]),
        "ema_26": exponential_moving_average(closes, DEFAULT_EMA_PERIODS[1]),
        "rsi_14": relative_strength_index(closes, DEFAULT_RSI_PERIOD),
        "macd": macd(closes, DEFAULT_MACD_FAST, DEFAULT_MACD_SLOW, DEFAULT_MACD_SIGNAL),
        "bollinger": bollinger_bands(closes, DEFAULT_BOLLINGER_PERIOD, DEFAULT_BOLLINGER_STD_DEV),
    }
