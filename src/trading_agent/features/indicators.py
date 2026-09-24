"""Aligned indicators; None denotes warm-up. Volatility is sample, not annualized."""

import math
from collections.abc import Sequence
from statistics import stdev

Series = list[float | None]


def _validate(values: Sequence[float], period: int = 1) -> None:
    if isinstance(period, bool) or not isinstance(period, int) or period < 1:
        raise ValueError("period must be a positive integer")
    if any(not math.isfinite(x) for x in values):
        raise ValueError("values must be finite")


def returns(values: Sequence[float]) -> list[float]:
    _validate(values)
    if any(x <= 0 for x in values):
        raise ValueError("prices must be positive")
    return [b / a - 1 for a, b in zip(values, values[1:], strict=False)]


def sma(values: Sequence[float], period: int) -> Series:
    _validate(values, period)
    return [
        None if i + 1 < period else sum(values[i + 1 - period : i + 1]) / period
        for i in range(len(values))
    ]


def ema(values: Sequence[float], period: int) -> list[float]:
    _validate(values, period)
    result: list[float] = []
    alpha = 2 / (period + 1)
    for value in values:
        result.append(value if not result else alpha * value + (1 - alpha) * result[-1])
    return result


def rolling_volatility(values: Sequence[float], period: int) -> Series:
    _validate(values, period)
    if period < 2:
        raise ValueError("sample volatility requires period >= 2")
    changes = returns(values)
    return [None if i < period else stdev(changes[i - period : i]) for i in range(len(values))]


def rsi(values: Sequence[float], period: int = 14) -> Series:
    _validate(values, period)
    returns(values)  # Validate positive prices.
    result: Series = [None] * len(values)
    if len(values) <= period:
        return result
    changes = [b - a for a, b in zip(values, values[1:], strict=False)]
    gain = sum(max(x, 0) for x in changes[:period]) / period
    loss = sum(max(-x, 0) for x in changes[:period]) / period
    for i in range(period, len(values)):
        if i > period:
            gain = (gain * (period - 1) + max(changes[i - 1], 0)) / period
            loss = (loss * (period - 1) + max(-changes[i - 1], 0)) / period
        result[i] = (50.0 if gain == 0 else 100.0) if loss == 0 else 100 - 100 / (1 + gain / loss)
    return result


def atr(
    high: Sequence[float], low: Sequence[float], close: Sequence[float], period: int = 14
) -> Series:
    for values in (high, low, close):
        _validate(values, period)
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("OHLC arrays must align")
    if any(not 0 < lo <= cl <= hi for hi, lo, cl in zip(high, low, close, strict=True)):
        raise ValueError("invalid OHLC prices")
    ranges = [
        max(hi - lo, abs(hi - close[i - 1]), abs(lo - close[i - 1])) if i else hi - lo
        for i, (hi, lo) in enumerate(zip(high, low, strict=True))
    ]
    result: Series = [None] * len(close)
    if len(close) >= period:
        value = sum(ranges[:period]) / period
        result[period - 1] = value
        for i in range(period, len(close)):
            value = (value * (period - 1) + ranges[i]) / period
            result[i] = value
    return result


def volume_ratio(values: Sequence[float], period: int) -> Series:
    """Current volume / mean of previous period volumes (excludes current bar)."""
    _validate(values, period)
    if any(x < 0 for x in values):
        raise ValueError("volume must be nonnegative")
    result: Series = []
    for i, value in enumerate(values):
        baseline = sum(values[i - period : i]) / period if i >= period else 0
        result.append(value / baseline if baseline else None)
    return result
