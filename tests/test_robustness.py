import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from decimal import Decimal

import pytest
from sqlalchemy import event, select
from sqlalchemy.exc import OperationalError
from test_execution import broker  # noqa: F401
from test_risk import NOW, order

from trading_agent.config.settings import Settings
from trading_agent.execution.paper import PaperBrokerAdapter
from trading_agent.models.domain import OrderRequest
from trading_agent.monitoring.cache import OptionalCache
from trading_agent.monitoring.logging import JSONFormatter
from trading_agent.persistence.store import Store
from trading_agent.persistence.tables import FillRecord, RiskRecord


def test_cache_fallback_and_expiration(monkeypatch):
    import trading_agent.monitoring.cache as module

    now = [100.0]
    monkeypatch.setattr(module, "monotonic", lambda: now[0])
    cache = OptionalCache("redis://127.0.0.1:1/0")
    cache.set("key", "value", ttl=2)
    assert cache.get("key") == "value"
    now[0] += 3
    assert cache.get("key") is None
    assert not cache.available()
    cache.close()


def test_logging_is_json():
    record = logging.LogRecord("test", logging.INFO, "", 0, "paper_order", (), None)
    record.order_id = "example"
    payload = json.loads(JSONFormatter().format(record))
    assert payload["event"] == "paper_order"
    assert payload["order_id"] == "example"


def test_configuration_rejects_live_mode(monkeypatch):
    monkeypatch.setenv("TRADING_EXECUTION_MODE", "live")
    with pytest.raises(ValueError):
        Settings()


def test_commit_failure_rolls_back_all_effects(broker):  # noqa: F811
    def fail_commit(connection):
        raise OperationalError("simulated", {}, Exception("database unavailable"))

    event.listen(broker.store.engine, "commit", fail_commit)
    with pytest.raises(OperationalError):
        broker.submit(order())
    event.remove(broker.store.engine, "commit", fail_commit)
    assert broker.portfolio().cash == Decimal("100000")
    with broker.store.sessions() as session:
        assert list(session.scalars(select(FillRecord))) == []
        assert list(session.scalars(select(RiskRecord))) == []
    assert broker.submit(order()).status == "filled"


def test_independent_adapters_share_atomic_limits(broker):  # noqa: F811
    settings = Settings(database_url=broker.settings.database_url, max_portfolio_exposure="1000")
    other = PaperBrokerAdapter(settings, Store(settings.database_url), broker.provider, lambda: NOW)
    first = PaperBrokerAdapter(settings, broker.store, broker.provider, lambda: NOW)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(adapter.submit, order(client_order_id=str(i)))
            for i, adapter in enumerate((first, other))
        ]
        results = [f.result() for f in futures]
    assert sum(x.status == "filled" for x in results) == 1
    assert broker.portfolio().exposure == Decimal("1000")


def test_rejected_order_id_cannot_be_reused(broker):  # noqa: F811
    assert broker.submit(order(stop_loss=None)).status == "rejected"
    assert "duplicate_order" in broker.submit(order()).risk.reasons


def test_constructed_invalid_order_cannot_bypass_validation(broker):  # noqa: F811
    bad = OrderRequest.model_construct(
        client_order_id="bad",
        symbol="DEMO",
        side="buy",
        quantity=-100,
        stop_loss=Decimal("90"),
        instrument_type="equity",
    )
    with pytest.raises(ValueError):
        broker.submit(bad)
    assert broker.portfolio().cash == Decimal("100000")


def test_future_portfolio_quote_fails_closed(broker):  # noqa: F811
    broker.submit(order())
    broker.provider.clock = lambda: NOW + timedelta(seconds=1)
    assert broker.submit(order(client_order_id="future")).status == "rejected"


def test_moving_clock_accepts_current_quotes(broker):  # noqa: F811
    tick = [NOW]

    def moving_clock():
        tick[0] += timedelta(microseconds=1)
        return tick[0]

    broker.clock = moving_clock
    broker.provider.clock = moving_clock
    assert broker.submit(order()).status == "filled"
    assert broker.portfolio().positions[0].quantity == 10
    assert broker.submit(order(client_order_id="next")).status == "filled"


def test_one_quote_per_symbol_for_risk_and_fill(broker):  # noqa: F811
    broker.submit(order(quantity=100))
    from trading_agent.models.domain import Instrument, Quote

    class ChangingProvider:
        calls = 0

        def quote(self, symbol):
            self.calls += 1
            return Quote(
                instrument=Instrument(symbol=symbol),
                price=Decimal("100") if self.calls == 1 else Decimal("200"),
                timestamp=NOW,
            )

    provider = ChangingProvider()
    adapter = PaperBrokerAdapter(
        Settings(max_portfolio_exposure="15000"), broker.store, provider, clock=lambda: NOW
    )
    result = adapter.submit(order(client_order_id="consistent", quantity=1))
    assert result.status == "filled"
    assert provider.calls == 1
    from trading_agent.persistence.tables import LedgerRecord

    with broker.store.sessions() as session:
        assert Decimal(session.get(LedgerRecord, 1).payload["exposure"]) <= Decimal("15000")


@pytest.mark.parametrize("url", ["sqlite:///:memory:", "sqlite://"])
def test_shared_connection_in_memory_databases_are_rejected(url):
    with pytest.raises(ValueError, match="file-backed"):
        Store(url)
