"""Offline synthetic research-to-risk-to-paper example; no external requests."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from trading_agent.agent.orchestrator import ResearchAgent
from trading_agent.config.settings import Settings
from trading_agent.execution.paper import PaperBrokerAdapter
from trading_agent.models.domain import Instrument, OHLCVBar, OrderRequest
from trading_agent.nse.mock import MockNSEProvider
from trading_agent.persistence.store import Store
from trading_agent.strategy.crossover import MovingAverageCrossover
from trading_agent.strategy.pipeline import ResearchPipeline


def main() -> None:
    now = datetime(2026, 9, 25, 5, 0, tzinfo=UTC)
    path = Path(__file__).resolve().parents[1] / ".tmp"
    path.mkdir(exist_ok=True)
    settings = Settings(database_url=f"sqlite:///{path}/demo.db")
    store = Store(settings.database_url)
    store.initialize(settings.initial_cash, now)
    provider = MockNSEProvider(clock=lambda: now)
    broker = PaperBrokerAdapter(settings, store, provider, clock=lambda: now)
    bars = [
        OHLCVBar(
            instrument=Instrument(symbol="DEMO"),
            timestamp=now - timedelta(minutes=4 - i),
            open=Decimal(p),
            high=Decimal(p),
            low=Decimal(p),
            close=Decimal(p),
            volume=100,
        )
        for i, p in enumerate([3, 2, 1, 4])
    ]
    pipeline = ResearchPipeline(MovingAverageCrossover(2, 3), ResearchAgent(), store)
    signal = pipeline.generate(bars, provider.market_context(), broker.portfolio())
    print(signal.model_dump_json(indent=2))
    if signal.side in ("buy", "sell"):
        result = broker.submit(
            OrderRequest(
                client_order_id="offline-demo-1",
                symbol=signal.symbol,
                side=signal.side,
                quantity=10,
                stop_loss=Decimal("90") if signal.side == "buy" else Decimal("110"),
            )
        )
        print(result.model_dump_json(indent=2))
    print(broker.portfolio().model_dump_json(indent=2))
    store.engine.dispose()


if __name__ == "__main__":
    main()
