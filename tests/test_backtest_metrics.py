from decimal import Decimal as D

from trading_agent.backtesting.metrics import compare_paper, compute_metrics


def test_metrics_known_drawdown_and_trade_outcomes() -> None:
    m = compute_metrics(
        [D("100"), D("110"), D("99")],
        [D("10"), D("-5")],
        periods_per_year=252,
        risk_free_rate=0,
        turnover=D("200"),
        exposures=[D("0"), D("50"), D("40")],
        costs=D("1"),
        slippage=D("2"),
    )
    assert abs(m["total_return"] + 0.01) < 1e-12
    assert abs(m["max_drawdown"] - 0.1) < 1e-12
    assert m["win_rate"] == 0.5
    assert m["profit_factor"] == 2
    assert m["expectancy"] == 2.5
    assert m["closed_trade_count"] == 2
    assert m["costs"] == 1
    assert compare_paper([D("100"), D("99")], [D("100"), D("101")])["final_equity_difference"] == -2


def test_flat_metrics_undefined_ratios() -> None:
    m = compute_metrics([D("100"), D("100")], [])
    assert m["sharpe"] is None
    assert m["calmar"] is None
    assert m["profit_factor"] is None


def test_short_runs_do_not_exponentiate_to_annual_returns() -> None:
    m = compute_metrics([D("100"), D("1000")], [], periods_per_year=100000)
    assert m["annualized_return"] is None


def test_bankruptcy_is_reported_without_dividing_by_zero() -> None:
    metrics = compute_metrics([D("100"), D("0"), D("0")], [D("-100")])
    assert metrics["total_return"] == -1
    assert metrics["max_drawdown"] == 1
