"""Opt-in, passwordless local socket integration tests; never target a broker."""

import os
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import URL
from test_risk import NOW, order

from trading_agent.config.settings import Settings
from trading_agent.execution.paper import PaperBrokerAdapter
from trading_agent.monitoring.cache import OptionalCache
from trading_agent.nse.mock import MockNSEProvider
from trading_agent.persistence.store import Store
from trading_agent.persistence.tables import FillRecord, RiskRecord


def test_postgres_transactions_restart_and_concurrency():
    socket = os.environ.get("TRADING_TEST_POSTGRES_SOCKET")
    if not socket:
        pytest.skip("set TRADING_TEST_POSTGRES_SOCKET for isolated local PostgreSQL test")
    assert Path(socket).resolve().is_relative_to(Path.cwd())
    base = URL.create(
        "postgresql+psycopg", username="paper", database="paper", query={"host": socket}
    )
    admin = create_engine(base)
    schema = "phase1_test_" + uuid4().hex
    with admin.begin() as connection:
        connection.execute(text(f"CREATE SCHEMA {schema}"))
    url = base.update_query_dict({"options": f"-csearch_path={schema}"}).render_as_string()
    stores = [Store(url), Store(url)]
    try:
        stores[0].initialize(Decimal("100000"), NOW)
        stores[1].initialize(Decimal("1"), NOW)  # Existing balance must survive initialization.
        settings = Settings(database_url=url, max_portfolio_exposure="1000")
        provider = MockNSEProvider(clock=lambda: NOW)
        adapters = [
            PaperBrokerAdapter(settings, store, provider, clock=lambda: NOW) for store in stores
        ]
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(adapters[i % 2].submit, order()) for i in range(8)]
            results = [future.result() for future in futures]
        assert sum(result.status == "filled" for result in results) == 1
        assert adapters[1].portfolio().cash == Decimal("99000")
        assert "portfolio_exposure" in adapters[1].submit(order(client_order_id="cap")).risk.reasons
        stores[1].engine.dispose()
        assert adapters[1].submit(order()).status == "rejected"
        with stores[0].sessions() as session:
            assert len(list(session.scalars(select(FillRecord)))) == 1
            assert len(list(session.scalars(select(RiskRecord)))) == 10
    finally:
        for store in stores:
            store.engine.dispose()
        with admin.begin() as connection:
            connection.execute(text(f"DROP SCHEMA {schema} CASCADE"))
        admin.dispose()


def test_valkey_real_cache():
    socket = os.environ.get("TRADING_TEST_VALKEY_SOCKET")
    if not socket:
        pytest.skip("set TRADING_TEST_VALKEY_SOCKET for isolated local Valkey test")
    assert Path(socket).resolve().is_relative_to(Path.cwd())
    cache = OptionalCache(f"unix://{socket}")
    assert cache.available()
    key = "phase1_test_" + uuid4().hex
    try:
        cache.set(key, "synthetic", ttl=1)
        assert cache.get(key) == "synthetic"
    finally:
        cache.close()
