from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from trading_agent.config.settings import Settings
from trading_agent.models.domain import Instrument, OrderRequest, PortfolioSnapshot, Position, Quote
from trading_agent.risk.engine import RiskEngine

NOW = datetime(2026, 9, 25, 5, 0, tzinfo=UTC)


def order(**updates):
    return OrderRequest(
        **(
            {
                "client_order_id": "test-1",
                "symbol": "DEMO",
                "side": "buy",
                "quantity": 10,
                "stop_loss": "90",
            }
            | updates
        )
    )


def quote(**updates):
    return Quote(
        **({"instrument": Instrument(symbol="DEMO"), "price": "100", "timestamp": NOW} | updates)
    )


def snapshot(**updates):
    return PortfolioSnapshot(
        **(
            {"cash": "100000", "equity": "100000", "day_start_equity": "100000", "as_of": NOW}
            | updates
        )
    )


def decide(o=None, q=None, p=None, settings=None, seen=None, now=NOW):
    return RiskEngine(settings or Settings()).evaluate(
        o or order(), q or quote(), p or snapshot(), seen or set(), now
    )


def test_approval():
    assert decide().approved


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"max_capital_per_trade": "999"}, "capital_per_trade"),
        ({"max_portfolio_exposure": "999"}, "portfolio_exposure"),
        ({"allowed_symbols": frozenset({"OTHER"})}, "symbol_not_allowed"),
        ({"allowed_instrument_types": frozenset({"future"})}, "instrument_type_not_allowed"),
        ({"holidays": frozenset({"2026-09-25"})}, "market_closed"),
    ],
)
def test_configurable_rejections(change, reason):
    result = decide(settings=Settings(**change))
    assert not result.approved
    assert reason in result.reasons


@pytest.mark.parametrize(
    ("q", "reason"),
    [
        (quote(timestamp=NOW - timedelta(seconds=31)), "stale_quote"),
        (quote(timestamp=NOW + timedelta(seconds=1)), "future_quote"),
        (quote(instrument=Instrument(symbol="OTHER")), "quote_mismatch"),
    ],
)
def test_quote_rejections(q, reason):
    assert reason in decide(q=q).reasons


@pytest.mark.parametrize("hour", [0, 3, 10, 20])
def test_closed_hours(hour):
    assert "market_closed" in decide(now=NOW.replace(hour=hour)).reasons


def test_weekend():
    assert "market_closed" in decide(now=NOW + timedelta(days=1)).reasons


def test_duplicate():
    assert "duplicate_order" in decide(seen={"test-1"}).reasons


@pytest.mark.parametrize("stop", [None, "100", "101"])
def test_stop_loss(stop):
    assert "invalid_stop_loss" in decide(o=order(stop_loss=stop)).reasons


def test_loss_threshold_includes_unrealized():
    assert "daily_loss" in decide(p=snapshot(equity="95000")).reasons
    assert decide(p=snapshot(equity="95000.01")).approved


def test_cash_and_overselling():
    assert "insufficient_cash" in decide(p=snapshot(cash="999")).reasons
    assert "insufficient_position" in decide(o=order(side="sell", stop_loss="110")).reasons


def test_position_limits_and_reductions():
    p = snapshot(
        positions=[Position(symbol="OTHER", quantity=1, average_price="100", mark_price="100")],
        exposure="100",
    )
    assert "open_positions" in decide(p=p, settings=Settings(max_open_positions=1)).reasons
    held = snapshot(
        positions=[Position(symbol="DEMO", quantity=10, average_price="100", mark_price="100")],
        exposure="1000",
    )
    assert decide(o=order(side="sell", stop_loss="110"), p=held).approved


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-1", "0"])
def test_invalid_money(bad):
    with pytest.raises(ValueError):
        quote(price=bad)


def test_domain_validation():
    with pytest.raises(ValueError):
        order(quantity=0)
    with pytest.raises(ValueError):
        quote(timestamp=NOW.replace(tzinfo=None))
    assert quote().price == Decimal("100")
