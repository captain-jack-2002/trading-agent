from contextlib import asynccontextmanager

from httpx import ASGITransport, AsyncClient
from test_risk import NOW, order

from trading_agent.api.app import create_app
from trading_agent.config.settings import Settings
from trading_agent.nse.mock import MockNSEProvider


@asynccontextmanager
async def client_for(app):
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            yield client


async def test_health_readiness_and_paper_routes(tmp_path):
    app = create_app(
        Settings(database_url=f"sqlite:///{tmp_path}/api.db"),
        provider=MockNSEProvider(clock=lambda: NOW),
        clock=lambda: NOW,
    )
    async with client_for(app) as client:
        assert (await client.get("/health")).json() == {"status": "ok", "execution_mode": "paper"}
        assert (await client.get("/ready")).status_code == 200
        result = await client.post("/paper/orders", json=order().model_dump(mode="json"))
        assert result.status_code == 200
        assert result.json()["status"] == "filled"
        assert (
            await client.post("/paper/orders", json=order().model_dump(mode="json"))
        ).status_code == 422
        assert (await client.get("/portfolio")).json()["cash"] == "99000"
        assert len((await client.get("/positions")).json()) == 1
        assert (await client.get("/risk/status")).json()["execution_mode"] == "paper"
        assert len((await client.get("/audit")).json()) == 2
        assert (await client.get("/signals")).json() == []
        assert (await client.post("/live/orders", json={})).status_code == 404
        assert (await client.post("/paper/orders", json={"approved": True})).status_code == 422


async def test_database_unavailable_degrades_readiness_and_orders(tmp_path):
    app = create_app(Settings(database_url=f"sqlite:///{tmp_path}/missing/none.db"))
    async with client_for(app) as client:
        assert (await client.get("/health")).status_code == 200
        assert (await client.get("/ready")).status_code == 503
        assert (
            await client.post("/paper/orders", json=order().model_dump(mode="json"))
        ).status_code == 503
        assert (await client.get("/portfolio")).status_code == 503


async def test_valkey_failure_is_optional(tmp_path):
    app = create_app(
        Settings(database_url=f"sqlite:///{tmp_path}/cache.db", valkey_url="redis://127.0.0.1:1/0")
    )
    async with client_for(app) as client:
        response = await client.get("/ready")
        assert response.status_code == 200
        assert response.json()["cache"] == "memory"
