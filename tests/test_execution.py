from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from decimal import Decimal

import pytest
from sqlalchemy import select
from test_risk import NOW, order

from trading_agent.config.settings import Settings
from trading_agent.execution.hdfc_sky import HDFCSkyAdapter
from trading_agent.execution.paper import PaperBrokerAdapter
from trading_agent.nse.mock import MockNSEProvider
from trading_agent.persistence.store import Store
from trading_agent.persistence.tables import FillRecord, OrderRecord, RiskRecord


@pytest.fixture
def broker(tmp_path):
    settings = Settings(database_url=f"sqlite:///{tmp_path}/paper.db")
    store = Store(settings.database_url)
    store.initialize(settings.initial_cash, NOW)
    provider = MockNSEProvider(clock=lambda: NOW)
    return PaperBrokerAdapter(settings, store, provider, clock=lambda: NOW)


def test_fill_and_partial_sell_accounting(broker):
    assert broker.submit(order()).status == "filled"
    broker.provider.set_price("DEMO", Decimal("120"))
    p = broker.portfolio()
    assert p.cash == Decimal("99000")
    assert p.unrealized_pnl == Decimal("200")
    result = broker.submit(order(client_order_id="sell", side="sell", quantity=4, stop_loss="130"))
    assert result.status == "filled"
    p = broker.portfolio()
    assert p.cash == Decimal("99480")
    assert p.realized_pnl == Decimal("80")
    assert p.unrealized_pnl == Decimal("120")
    assert p.equity == Decimal("100200")
    assert p.positions[0].quantity == 6


def test_weighted_cost_and_full_close(broker):
    broker.submit(order())
    broker.provider.set_price("DEMO", Decimal("120"))
    broker.submit(order(client_order_id="second"))
    assert broker.portfolio().positions[0].average_price == Decimal("110")
    broker.submit(order(client_order_id="sell", side="sell", quantity=20, stop_loss="130"))
    assert broker.portfolio().positions == ()
    assert broker.portfolio().realized_pnl == Decimal("200")


def test_direct_adapter_rejection_cannot_fill(broker):
    result = broker.submit(order(stop_loss=None))
    assert result.status == "rejected"
    assert broker.portfolio().cash == Decimal("100000")
    with broker.store.sessions() as session:
        assert list(session.scalars(select(FillRecord))) == []
        assert len(list(session.scalars(select(RiskRecord)))) == 1


def test_concurrent_duplicates_and_restart(broker):
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: broker.submit(order()), range(8)))
    assert sum(x.status == "filled" for x in results) == 1
    restored_store = Store(broker.settings.database_url)
    restored_store.initialize(broker.settings.initial_cash, NOW)
    restored = PaperBrokerAdapter(
        broker.settings, restored_store, broker.provider, clock=lambda: NOW
    )
    assert restored.submit(order()).status == "rejected"
    assert restored.portfolio().positions[0].quantity == 10
    with broker.store.sessions() as session:
        assert len(list(session.scalars(select(OrderRecord)))) == 1


def test_unrealized_daily_loss_blocks(broker):
    broker.submit(order(quantity=100))
    broker.provider.set_price("DEMO", Decimal("40"))
    assert "daily_loss" in broker.submit(order(client_order_id="loss", stop_loss="30")).risk.reasons


def test_stale_held_quote_blocks_other_order(broker):
    broker.submit(order())
    broker.provider.clock = lambda: NOW - timedelta(minutes=2)
    result = broker.submit(order(client_order_id="stale"))
    assert not result.risk.approved
    assert "invalid_portfolio_quote" in result.risk.reasons


def test_unknown_symbol_is_audited_rejection(broker):
    assert broker.submit(order(symbol="UNKNOWN")).status == "rejected"


def test_live_adapter_is_disabled():
    with pytest.raises(NotImplementedError):
        HDFCSkyAdapter().submit(order())


def test_day_rollover_preserves_overnight_gap(broker):
    broker.submit(order())
    next_day = NOW + timedelta(days=3)
    broker.clock = lambda: next_day
    broker.provider.clock = lambda: next_day
    broker.provider.set_price("DEMO", Decimal("90"))
    p = broker.portfolio()
    assert p.daily_pnl == Decimal("-100")
    assert p.trading_day == next_day.date()
