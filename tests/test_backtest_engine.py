from datetime import date, datetime, timedelta
from decimal import Decimal as D
from pathlib import Path

import pytest

from trading_agent.backtesting import BacktestConfig, BacktestEngine, SimulationOrder
from trading_agent.config.settings import Settings
from trading_agent.data.calendar import MarketCalendar, TradingSession
from trading_agent.data.schemas.canonical import CanonicalBar
from trading_agent.models.domain import OrderRequest

START = datetime.fromisoformat("2026-09-24T09:15:00+05:30")


def bars() -> list[CanonicalBar]:
    return [
        CanonicalBar(
            asset_class="equity",
            symbol="DEMO",
            exchange="NSE",
            timestamp=START + timedelta(minutes=i + 1),
            open=D(p),
            high=D(p) + 1,
            low=D(p) - 1,
            close=D(p),
            volume=100,
        )
        for i, p in enumerate(["100", "110", "120"])
    ]


def calendar() -> MarketCalendar:
    return MarketCalendar(
        sessions=(
            TradingSession(
                session_date=date(2026, 9, 24), opens_at=START, closes_at=START + timedelta(hours=6)
            ),
        )
    )


def order(id: str = "one", side: str = "buy") -> SimulationOrder:
    return SimulationOrder(
        order=OrderRequest(
            client_order_id=id,
            symbol="DEMO",
            side=side,
            quantity=1,
            stop_loss=D("90") if side == "buy" else D("150"),
        )
    )  # type: ignore[arg-type]


def test_next_open_closed_history_and_accounting(tmp_path: Path) -> None:
    histories: list[int] = []

    def strategy(history: tuple[CanonicalBar, ...]) -> list[SimulationOrder]:
        histories.append(len(history))
        return (
            [order()] if len(history) == 1 else [order("two", "sell")] if len(history) == 2 else []
        )

    result = BacktestEngine(Settings(), BacktestConfig(equity_tick_size=D(".05")), calendar()).run(
        bars(), strategy
    )
    assert histories == [1, 2, 3]
    assert [t.fill_price for t in result.trades] == [D("110"), D("120")]
    assert result.final_portfolio.cash == D("100010")
    assert result.final_portfolio.realized_pnl == D("10")
    assert result.metrics["closed_trade_count"] == 1
    result.write_reports(tmp_path)
    assert "SYNTHETIC" in (tmp_path / "report.md").read_text()


def test_calendar_risk_duplicate_and_chronology() -> None:
    engine = BacktestEngine(Settings(), BacktestConfig(equity_tick_size=D(".05")), MarketCalendar())
    result = engine.run(bars(), lambda h: [order()])
    assert not result.trades
    assert "calendar_closed" in result.audit[0].reasons
    with pytest.raises(ValueError, match="chronolog"):
        engine.run(list(reversed(bars())), lambda h: [])
    result = BacktestEngine(Settings(), BacktestConfig(equity_tick_size=D(".05")), calendar()).run(
        bars(), lambda h: [order()]
    )
    assert len(result.trades) == 1
    assert any("duplicate_order" in a.reasons for a in result.audit)


def test_limit_no_intrabar_lookahead_and_no_limit_violation() -> None:
    limit = order().model_copy(update={"order_type": "limit", "limit_price": D("109.5")})
    result = BacktestEngine(Settings(), BacktestConfig(equity_tick_size=D(".05")), calendar()).run(
        bars(), lambda h: [limit]
    )
    assert not result.trades
    assert "limit_not_marketable_at_open" in result.audit[0].reasons


def test_recheck_slippage_and_cost_affordability() -> None:
    from trading_agent.backtesting.costs import CostSchedule, Rate, Slippage

    config = BacktestConfig(equity_tick_size=D(".05"), slippage=Slippage(percentage=D(".2")))
    result = BacktestEngine(Settings(max_capital_per_trade=D("115")), config, calendar()).run(
        bars(), lambda h: [order()]
    )
    assert not result.trades
    assert "capital_per_trade" in result.audit[0].reasons
    config = BacktestConfig(
        equity_tick_size=D(".05"),
        costs={"equity_delivery": CostSchedule(brokerage=Rate(buy=D(".1")))},
    )
    result = BacktestEngine(Settings(initial_cash=D("115")), config, calendar()).run(
        bars(), lambda h: [order()]
    )
    assert not result.trades
    assert "insufficient_cash_with_costs" in result.audit[0].reasons


def test_invalid_signals_fail_closed() -> None:
    result = BacktestEngine(Settings(), BacktestConfig(equity_tick_size=D(".05")), calendar()).run(
        bars(), lambda h: [object()]
    )  # type: ignore[list-item]
    assert not result.trades
    assert result.audit[0].reasons == ("invalid_strategy_signal",)


def test_open_execution_never_uses_future_bar_volume() -> None:
    source = bars()
    changed = [source[0], source[1].model_copy(update={"volume": 0}), source[2]]

    def strategy(h: tuple[CanonicalBar, ...]) -> list[SimulationOrder]:
        return [order()] if len(h) == 1 else []

    engine = BacktestEngine(Settings(), BacktestConfig(equity_tick_size=D(".05")), calendar())
    first = engine.run(source, strategy)
    second = engine.run(changed, strategy)
    assert first.trades == second.trades
    assert first.trades[0].timestamp == source[0].timestamp


def test_rounding_limit_and_partial_rejections() -> None:
    from trading_agent.backtesting.costs import Slippage

    config = BacktestConfig(equity_tick_size=D(".05"), slippage=Slippage(bps=D("1")))
    result = BacktestEngine(Settings(), config, calendar()).run(
        bars(), lambda h: [order()] if len(h) == 1 else []
    )
    assert result.trades[0].fill_price == D("110.05")
    capped = order().model_copy(update={"order_type": "limit", "limit_price": D("110")})
    result = BacktestEngine(Settings(), config, calendar()).run(bars(), lambda h: [capped])
    assert not result.trades
    oversized = SimulationOrder(order=order().order.model_copy(update={"quantity": 101}))
    result = BacktestEngine(Settings(), BacktestConfig(equity_tick_size=D(".05")), calendar()).run(
        bars(), lambda h: [oversized]
    )
    assert "partial_fill_unsupported" in result.audit[0].reasons


def test_cost_basis_reconciles_realized_unrealized_and_cash() -> None:
    from trading_agent.backtesting.costs import CostSchedule, Rate

    config = BacktestConfig(
        equity_tick_size=D(".05"),
        costs={"equity_delivery": CostSchedule(brokerage=Rate(buy=D(".01"), sell=D(".01")))},
    )
    result = BacktestEngine(Settings(), config, calendar()).run(
        bars(),
        lambda h: [order()] if len(h) == 1 else [order("sell", "sell")] if len(h) == 2 else [],
    )
    assert result.final_portfolio.realized_pnl == D("7.7")
    assert result.final_portfolio.cash == D("100007.7")
    assert result.metrics["costs"] == 2.3


def test_phase1_session_and_stop_policy_stay_authoritative() -> None:
    weekend = [b.model_copy(update={"timestamp": b.timestamp + timedelta(days=2)}) for b in bars()]
    special = MarketCalendar(
        sessions=(
            TradingSession(
                session_date=date(2026, 9, 26),
                opens_at=START + timedelta(days=2),
                closes_at=START + timedelta(days=2, hours=6),
                special=True,
            ),
        )
    )
    result = BacktestEngine(Settings(), BacktestConfig(equity_tick_size=D(".05")), special).run(
        weekend, lambda h: [order()]
    )
    assert not result.trades
    assert "market_closed" in result.audit[0].reasons
    no_stop = SimulationOrder(order=order().order.model_copy(update={"stop_loss": None}))
    result = BacktestEngine(Settings(), BacktestConfig(equity_tick_size=D(".05")), calendar()).run(
        bars(), lambda h: [no_stop]
    )
    assert not result.trades
    assert "invalid_stop_loss" in result.audit[0].reasons


def test_partial_fill_and_reject_model() -> None:
    request = SimulationOrder(order=order().order.model_copy(update={"quantity": 10}))
    result = BacktestEngine(
        Settings(),
        BacktestConfig(equity_tick_size=D(".01"), fill_mode="partial", fill_fraction=D(".25")),
        calendar(),
    ).run(bars(), lambda h: [request] if len(h) == 1 else [])
    assert result.trades[0].order.quantity == 2
    assert result.audit[0].status == "partially_filled"
    result = BacktestEngine(
        Settings(), BacktestConfig(equity_tick_size=D(".01"), fill_mode="reject"), calendar()
    ).run(bars(), lambda h: [request])
    assert not result.trades
    assert "fill_model_reject" in result.audit[0].reasons


def test_report_never_overwrites(tmp_path: Path) -> None:
    result = BacktestEngine(Settings(), BacktestConfig(equity_tick_size=D(".01")), calendar()).run(
        bars(), lambda h: []
    )
    result.write_reports(tmp_path)
    with pytest.raises(FileExistsError):
        result.write_reports(tmp_path)
    assert "database_url" not in result.risk_settings


def test_derivative_contract_lots_fully_funded_and_product() -> None:
    from trading_agent.data.schemas.canonical import Contract

    contract = Contract(
        contract_id="SYNTH-FUT",
        instrument_type="FUTSTK",
        symbol="DEMO",
        exchange="NSE",
        underlying="DEMO",
        expiry=date(2026, 10, 1),
        lot_size=5,
        tick_size=D(".05"),
    )
    source = [
        CanonicalBar(
            asset_class="future",
            symbol="DEMO",
            exchange="NSE",
            timestamp=b.timestamp,
            open=b.open,
            high=b.high,
            low=b.low,
            close=b.close,
            volume=100,
            underlying="DEMO",
            expiry=contract.expiry,
            contract=contract,
        )
        for b in bars()
    ]
    settings = Settings(allowed_instrument_types=frozenset({"future"}))

    def request(quantity: int, product: str = "future") -> SimulationOrder:
        return SimulationOrder(
            order=OrderRequest(
                client_order_id="fut",
                symbol="DEMO",
                instrument_type="future",
                side="buy",
                quantity=quantity,
                stop_loss=D("90"),
            ),
            product=product,
        )  # type: ignore[arg-type]

    engine = BacktestEngine(settings, BacktestConfig(equity_tick_size=D(".01")), calendar())
    result = engine.run(source, lambda h: [request(5)] if len(h) == 1 else [])
    assert result.final_portfolio.cash == D("99450")
    result = engine.run(source, lambda h: [request(1)])
    assert not result.trades
    assert "invalid_lot_quantity" in result.audit[0].reasons
    result = engine.run(source, lambda h: [request(5, "equity_delivery")])
    assert not result.trades
    assert "cost_product_mismatch" in result.audit[0].reasons


def test_daily_loss_and_exposure_block_further_orders() -> None:
    source = bars()
    source[1] = source[1].model_copy(update={"low": D("90"), "close": D("90")})
    engine = BacktestEngine(
        Settings(max_daily_loss=D("10")), BacktestConfig(equity_tick_size=D(".01")), calendar()
    )
    result = engine.run(source, lambda h: [order(str(len(h)))])
    # The next open is 120 and recovers the loss before risk evaluation.
    assert len(result.trades) == 2
    source[2] = source[2].model_copy(
        update={"open": D("90"), "low": D("89"), "high": D("91"), "close": D("90")}
    )
    result = engine.run(source, lambda h: [order(str(len(h)))])
    assert len(result.trades) == 1
    assert "daily_loss" in result.audit[1].reasons
    engine = BacktestEngine(
        Settings(max_portfolio_exposure=D("150")),
        BacktestConfig(equity_tick_size=D(".01")),
        calendar(),
    )
    result = engine.run(bars(), lambda h: [order(str(len(h)))])
    assert len(result.trades) == 1
    assert "portfolio_exposure" in result.audit[1].reasons


def test_report_config_fixture_loading() -> None:
    for name in ("zero_costs.json", "illustrative_costs.json"):
        config = BacktestConfig.model_validate_json((Path("examples/backtest") / name).read_text())
        assert config.synthetic
        assert set(config.costs) == {"equity_delivery", "equity_intraday", "future", "option"}
        assert "SYNTHETIC" in config.provenance


def test_same_open_fills_share_historical_volume_capacity() -> None:
    requests = [
        SimulationOrder(order=order(str(i)).order.model_copy(update={"quantity": 75}))
        for i in range(2)
    ]
    result = BacktestEngine(Settings(), BacktestConfig(equity_tick_size=D(".01")), calendar()).run(
        bars(), lambda h: requests if len(h) == 1 else []
    )
    assert len(result.trades) == 1
    assert "partial_fill_unsupported" in result.audit[1].reasons


def test_fill_slippage_does_not_remark_existing_holdings() -> None:
    from trading_agent.backtesting.costs import Slippage
    from trading_agent.backtesting.engine import Ledger

    ledger = Ledger(D("100000"))
    initial = order().order.model_copy(update={"quantity": 50})
    ledger.fill(initial, D("100"), D("0"))
    request = SimulationOrder(
        order=initial.model_copy(update={"client_order_id": "second", "quantity": 10})
    )
    engine = BacktestEngine(
        Settings(max_portfolio_exposure=D("6100")),
        BacktestConfig(equity_tick_size=D(".01"), slippage=Slippage(percentage=D(".1"))),
        calendar(),
    )
    source = bars()[0].model_copy(update={"timestamp": START})
    trade, _ = engine._execute(request, source, ledger, set(), [bars()[0]], None)
    assert trade is not None
    assert ledger.snapshot(START).exposure == D("6000")


def test_backtest_warmup_history_is_available_without_warmup_trading() -> None:
    complete = bars()
    history_seen = []
    result = BacktestEngine(Settings(), BacktestConfig(equity_tick_size=D(".01")), calendar()).run(
        complete[1:],
        lambda history: history_seen.append(len(history)) or [],
        warmup_bars=complete[:1],
    )
    assert history_seen[0] == 2
    assert result.start == complete[1].timestamp - timedelta(seconds=60)
    assert len(result.equity) == len(complete)


def test_backtest_rejects_warmup_bar_after_first_execution_open() -> None:
    complete = bars()
    with pytest.raises(ValueError, match="warmup"):
        BacktestEngine(Settings(), BacktestConfig(equity_tick_size=D(".01")), calendar()).run(
            complete[1:], lambda history: [], warmup_bars=complete[1:2]
        )
