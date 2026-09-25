import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from trading_agent.config.settings import Settings
from trading_agent.execution.paper import PaperBrokerAdapter
from trading_agent.integrations.nse_mcp import (
    NSEMCPIntegration,
    NSEMCPProvider,
    config_from_settings,
)
from trading_agent.market_data.interfaces import NSEDataProvider
from trading_agent.models.domain import OrderRequest, PortfolioSnapshot, Position
from trading_agent.monitoring.cache import OptionalCache
from trading_agent.monitoring.logging import configure_logging
from trading_agent.nse.mock import MockNSEProvider, utc_now
from trading_agent.persistence.store import Store
from trading_agent.persistence.tables import SignalRecord

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    provider: NSEDataProvider | None = None,
    clock: Callable[[], datetime] = utc_now,
    nse_mcp: NSEMCPIntegration | None = None,
) -> FastAPI:
    config = settings or Settings()
    store = Store(config.database_url)
    data = provider or MockNSEProvider(clock)
    broker = PaperBrokerAdapter(config, store, data, clock)
    cache = OptionalCache(config.valkey_url)
    nse_mcp_integration = nse_mcp or NSEMCPIntegration(config_from_settings(config))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        try:
            store.initialize(config.initial_cash, clock())
        except SQLAlchemyError:
            logger.error("database_initialization_failed")
        yield
        cache.close()
        store.engine.dispose()

    app = FastAPI(title="Trading Agent — paper only", version="0.1.0", lifespan=lifespan)
    app.state.nse_research_provider = NSEMCPProvider(nse_mcp_integration, cache)

    @app.exception_handler(SQLAlchemyError)
    async def database_error(request: Any, exc: SQLAlchemyError) -> JSONResponse:
        logger.error("database_operation_failed")
        return JSONResponse(
            status_code=503, content={"detail": "Database unavailable; no order confirmed"}
        )

    def require_ready() -> None:
        if not store.ready():
            raise HTTPException(503, "Paper ledger unavailable")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "execution_mode": "paper"}

    @app.get("/ready")
    def ready() -> JSONResponse:
        database = store.ready()
        return JSONResponse(
            status_code=200 if database else 503,
            content={
                "status": "ready" if database else "not_ready",
                "database": database,
                "cache": "valkey" if cache.available() else "memory",
                "data_source": "mock",
            },
        )

    @app.get("/portfolio")
    def portfolio() -> PortfolioSnapshot:
        require_ready()
        try:
            return broker.portfolio()
        except (LookupError, ValueError) as exc:
            raise HTTPException(503, "Portfolio marks unavailable") from exc

    @app.get("/positions")
    def positions() -> tuple[Position, ...]:
        return portfolio().positions

    @app.get("/signals")
    def signals(limit: int = Query(default=100, ge=1, le=1000)) -> list[dict[str, Any]]:
        require_ready()
        with store.sessions() as session:
            return [
                row.payload
                for row in session.scalars(
                    select(SignalRecord).order_by(SignalRecord.id.desc()).limit(limit)
                )
            ]

    @app.get("/risk/status")
    def risk_status() -> dict[str, Any]:
        p = portfolio()
        context = cache.get("market_context")
        if context is None:
            context = data.market_context().get("description", "No context")
            cache.set("market_context", context)
        return {
            "execution_mode": "paper",
            "daily_pnl": str(p.daily_pnl),
            "daily_loss_blocked": p.daily_pnl <= -config.max_daily_loss,
            "max_daily_loss": str(config.max_daily_loss),
            "max_capital_per_trade": str(config.max_capital_per_trade),
            "max_portfolio_exposure": str(config.max_portfolio_exposure),
            "max_open_positions": config.max_open_positions,
            "allowed_symbols": sorted(config.allowed_symbols),
            "allowed_instrument_types": sorted(config.allowed_instrument_types),
            "mandatory_stop_loss": config.mandatory_stop_loss,
            "quote_max_age_seconds": config.quote_max_age_seconds,
            "market_context": context,
        }

    @app.get("/nse-mcp/status")
    def nse_mcp_status() -> dict[str, Any]:
        return asyncio.run(nse_mcp_integration.status())

    @app.get("/nse-mcp/tools")
    def nse_mcp_tools() -> dict[str, Any]:
        async def discover() -> dict[str, Any]:
            async def list_server(operation: Any) -> dict[str, Any]:
                try:
                    tools = await operation()
                    return {
                        "status": "available",
                        "tools": [tool.model_dump(mode="json") for tool in tools],
                    }
                except Exception as exc:
                    return {"status": "unavailable", "error": type(exc).__name__, "tools": []}

            bhavcopy, cm_market = await asyncio.gather(
                list_server(nse_mcp_integration.discover_bhavcopy_tools),
                list_server(nse_mcp_integration.discover_cm_market_tools),
            )
            return {"bhavcopy": bhavcopy, "cm_market": cm_market}

        return asyncio.run(discover())

    @app.post("/paper/orders")
    def paper_order(order: OrderRequest) -> JSONResponse:
        require_ready()
        result = broker.submit(order)
        return JSONResponse(
            status_code=200 if result.status == "filled" else 422,
            content=result.model_dump(mode="json"),
        )

    @app.get("/audit")
    def audit(limit: int = Query(default=100, ge=1, le=1000)) -> list[dict[str, Any]]:
        require_ready()
        return store.audit(limit)

    return app


app = create_app()
