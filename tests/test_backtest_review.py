"""Independent review regressions for aggregate liquidity and terminal insolvency."""

from decimal import Decimal as D

from test_backtest_engine import bars, calendar, order

from trading_agent.backtesting import BacktestConfig, BacktestEngine, SimulationOrder
from trading_agent.backtesting.costs import CostSchedule, Rate
from trading_agent.config.settings import Settings


def test_review_capacity_is_shared_by_orders_at_one_open() -> None:
    source = bars()
    requests = [
        SimulationOrder(order=order(name).order.model_copy(update={"quantity": 75}))
        for name in ("first", "second")
    ]
    result = BacktestEngine(Settings(), BacktestConfig(equity_tick_size=D(".01")), calendar()).run(
        source, lambda h: requests if len(h) == 1 else []
    )
    assert sum(t.order.quantity for t in result.trades) <= source[0].volume


def test_review_terminal_zero_equity_is_reportable() -> None:
    config = BacktestConfig(
        equity_tick_size=D(".01"),
        costs={"equity_delivery": CostSchedule(brokerage=Rate(sell=D("1")))},
    )
    result = BacktestEngine(Settings(initial_cash=D("110")), config, calendar()).run(
        bars(),
        lambda h: [order()] if len(h) == 1 else [order("exit", "sell")] if len(h) == 2 else [],
    )
    assert result.final_portfolio.equity == D("0")
    assert result.metrics["total_return"] == -1
    assert result.metrics["max_drawdown"] == 1
