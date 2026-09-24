import math

import pytest

from trading_agent.features.indicators import (
    atr,
    ema,
    returns,
    rolling_volatility,
    rsi,
    sma,
    volume_ratio,
)


def test_returns_and_averages():
    assert returns([100, 110, 99]) == pytest.approx([0.1, -0.1])
    assert sma([1, 2, 3, 4], 2) == [None, 1.5, 2.5, 3.5]
    assert ema([1, 2, 3], 2) == pytest.approx([1, 5 / 3, 23 / 9])


def test_volatility_rsi_atr_volume():
    assert rolling_volatility([100, 110, 99], 2)[-1] == pytest.approx(math.sqrt(0.02))
    assert rsi([1, 2, 3, 4], 2) == [None, None, 100, 100]
    assert rsi([4, 3, 2], 2)[-1] == 0
    assert rsi([2, 2, 2], 2)[-1] == 50
    assert atr([11, 13, 14], [9, 10, 12], [10, 12, 13], 2) == [None, 2.5, 2.25]
    assert volume_ratio([10, 10, 20], 2) == [None, None, 2]
    assert volume_ratio([0, 0, 1], 2)[-1] is None


@pytest.mark.parametrize("fn", [sma, ema, rolling_volatility, rsi, volume_ratio])
def test_bad_period(fn):
    with pytest.raises(ValueError):
        fn([1, 2], 0)


@pytest.mark.parametrize("values", [[1, float("nan")], [float("inf")]])
def test_nonfinite(values):
    with pytest.raises(ValueError):
        sma(values, 2)


def test_empty_and_invalid_prices():
    assert returns([]) == []
    assert sma([], 2) == []
    with pytest.raises(ValueError):
        returns([0, 1])
    with pytest.raises(ValueError):
        atr([1], [], [], 2)
