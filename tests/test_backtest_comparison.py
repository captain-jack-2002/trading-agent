from decimal import Decimal as D

import pytest

from trading_agent.backtesting.comparison import RecordedTrade, compare_trades


def test_recorded_trade_comparison_matches_ids_and_reports_missing() -> None:
    simulated = [
        RecordedTrade(client_order_id="same", quantity=10, price=D("100"), fees=D("2")),
        RecordedTrade(client_order_id="sim-only", quantity=1, price=D("20")),
    ]
    paper = [
        RecordedTrade(client_order_id="same", quantity=9, price=D("101"), fees=D("3")),
        RecordedTrade(client_order_id="paper-only", quantity=1, price=D("30")),
    ]
    result = compare_trades(simulated, paper)
    assert result.matched[0].price_difference == D("-1")
    assert result.matched[0].quantity_difference == 1
    assert result.matched[0].fee_difference == D("-1")
    assert result.only_simulated == ("sim-only",)
    assert result.only_paper == ("paper-only",)
    with pytest.raises(ValueError, match="duplicate"):
        compare_trades([simulated[0], simulated[0]], paper)
