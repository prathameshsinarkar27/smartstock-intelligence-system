"""
technical_indicators.py

Computes standard technical analysis indicators — SMA, EMA, RSI, MACD, and
Bollinger Bands — from a company's historical price data.

All indicator periods are configurable via function parameters; defaults
follow the conventional values used across most charting platforms:
    SMA:        20-day and 50-day
    EMA:        12-day and 26-day
    RSI:        14-day
    MACD:       12/26/9 (fast EMA, slow EMA, signal EMA)
    Bollinger:  20-day SMA basis, 2 standard deviations

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
    Compute the Simple Moving Average (SMA) over a rolling window.

    Args:
        closes: Closing prices, oldest first.
        period: Window size in days, e.g. 20 for a 20-day SMA.

    Returns:
        A list the same length as `closes`, with the first `period - 1`
        entries as None (not enough data yet to compute a full window) and
        the rest as the rolling mean of the trailing `period` closes.
        Returns an all-None list if `closes` has fewer than `period` entries.
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
    Compute the Exponential Moving Average (EMA), which weights recent
    prices more heavily than older ones.

    The first valid EMA value is seeded with a Simple Moving Average over
    the first `period` closes (the standard convention), after which each
    subsequent value is computed with the smoothing formula:
        EMA[i] = close[i] * k + EMA[i-1] * (1 - k),  where k = 2 / (period + 1)

    Args:
        closes: Closing prices, oldest first.
        period: Smoothing window in days, e.g. 12 for a 12-day EMA.

    Returns:
        A list the same length as `closes`, with the first `period - 1`
        entries as None, and the rest as the EMA value at that index.
        Returns an all-None list if `closes` has fewer than `period` entries.
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
    Compute the Relative Strength Index (RSI), a momentum oscillator
    ranging from 0-100. Conventionally, RSI > 70 suggests an asset may be
    overbought and RSI < 30 suggests it may be oversold — see
    `interpret_rsi()` for how this module labels those thresholds.

    Uses Wilder's smoothing method (the standard RSI convention): the
    first average gain/loss is a simple average over the first `period`
    day-over-day changes, and subsequent averages are smoothed using the
    same exponential-style formula Wilder originally defined.

    Args:
        closes: Closing prices, oldest first.
        period: Lookback window in days, conventionally 14.

    Returns:
        A list the same length as `closes`, with the first `period`
        entries as None (RSI needs `period` day-over-day changes, which
        requires `period + 1` prices), and the rest as the RSI value
        (0-100) at that index. Returns an all-None list if `closes` has
        fewer than `period + 1` entries.
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

    # changes[period - 1] is the day-over-day change ending at closes[period],
    # so the first computable RSI value aligns with index `period` in closes.
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
    Compute MACD (Moving Average Convergence Divergence): the difference
    between a fast and slow EMA (the "MACD line"), an EMA of that
    difference (the "signal line"), and the difference between the two
    (the "histogram") — used to spot momentum shifts.

    Args:
        closes: Closing prices, oldest first.
        fast_period: Fast EMA window, conventionally 12.
        slow_period: Slow EMA window, conventionally 26.
        signal_period: Signal line EMA window, conventionally 9.

    Returns:
        A dict with three lists, each the same length as `closes`:
            - "macd_line": fast EMA minus slow EMA.
            - "signal_line": EMA of the MACD line.
            - "histogram": macd_line minus signal_line.
        Entries are None wherever there isn't enough data yet for that
        particular series (the signal line and histogram start later than
        the MACD line, since the signal line is itself a derived EMA).
    """
    fast_ema = exponential_moving_average(closes, fast_period)
    slow_ema = exponential_moving_average(closes, slow_period)

    macd_line: list[float | None] = [
        (f - s) if f is not None and s is not None else None
        for f, s in zip(fast_ema, slow_ema)
    ]

    # The signal line is an EMA of the MACD line itself, computed only over
    # the contiguous non-None tail of macd_line (everything before the
    # slow EMA has converged is undefined, not just "missing").
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
    Compute Bollinger Bands: a moving average ("middle band") plus and
    minus a multiple of the rolling standard deviation ("upper"/"lower"
    bands) — used to visualize volatility and potential overbought/
    oversold conditions relative to recent price action.

    Args:
        closes: Closing prices, oldest first.
        period: Window size for the moving average and std dev, conventionally 20.
        num_std_dev: Number of standard deviations for the upper/lower bands,
            conventionally 2.

    Returns:
        A dict with three lists, each the same length as `closes`:
            "middle_band" (the SMA), "upper_band", "lower_band". Entries
            are None for the first `period - 1` indices, same as
            simple_moving_average().
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
    Translate the latest RSI value into a conventional signal label, for
    display on the Company Detail page.

    Args:
        latest_rsi: The most recent RSI value (0-100), or None if not
            enough data exists to compute one yet.

    Returns:
        "Overbought" if RSI > 70, "Oversold" if RSI < 30, "Neutral"
        otherwise, or "—" if latest_rsi is None.
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
    Translate the latest MACD line vs. signal line relationship into a
    conventional signal label.

    Args:
        latest_macd_line: The most recent MACD line value, or None.
        latest_signal_line: The most recent signal line value, or None.

    Returns:
        "Bullish" if the MACD line is above the signal line, "Bearish" if
        below, "—" if either value is unavailable yet.
    """
    if latest_macd_line is None or latest_signal_line is None:
        return "—"
    return "Bullish" if latest_macd_line > latest_signal_line else "Bearish"


def compute_all_indicators(closes: list[float]) -> dict[str, object]:
    """
    Compute the full suite of technical indicators for a price series, all
    at default/conventional periods, in one call — the shape the Company
    Detail page's chart-data endpoint and template need.

    Args:
        closes: Closing prices, oldest first.

    Returns:
        A dict with keys: sma_20, sma_50, ema_12, ema_26, rsi_14, macd
        (a dict with macd_line/signal_line/histogram), bollinger (a dict
        with middle_band/upper_band/lower_band). Each series is a list the
        same length as `closes`, with leading Nones where insufficient
        data exists, per each function's own docstring above.
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
