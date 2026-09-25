from decimal import Decimal as D

import pytest

from trading_agent.backtesting.costs import CostSchedule, Rate, Slippage


def test_side_specific_cost_bases_cap_and_tax() -> None:
    schedule = CostSchedule(
        brokerage=Rate(buy=D(".01"), sell=D(".01"), cap=D("2")),
        stt=Rate(sell=D(".001")),
        gst_rate=D(".18"),
    )
    buy = schedule.calculate("buy", D("100"), 10)
    sell = schedule.calculate("sell", D("100"), 10)
    assert buy.total == D("2.36")
    assert sell.total == D("3.36")
    assert sell.stt == D("1")


def test_zero_and_invalid_costs() -> None:
    assert CostSchedule().calculate("buy", D("10"), 1).total == 0
    with pytest.raises(ValueError):
        Rate(buy=D("-1"))
    with pytest.raises(ValueError):
        Slippage(bps=D("NaN"))


def test_slippage_adverse() -> None:
    slip = Slippage(bps=D("10"), percentage=D(".01"))
    assert slip.price(D("100"), "buy") == D("101.100")
    assert slip.price(D("100"), "sell") == D("98.900")
