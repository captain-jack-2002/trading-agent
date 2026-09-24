from collections.abc import Sequence

from trading_agent.agent.orchestrator import AgentOrchestrator
from trading_agent.models.domain import OHLCVBar, PortfolioSnapshot, TradeSignal
from trading_agent.persistence.store import Store
from trading_agent.persistence.tables import AuditRecord, ObservationRecord, SignalRecord
from trading_agent.strategy.base import Strategy


class ResearchPipeline:
    """Persist deterministic research; callers explicitly size and submit proposals separately."""

    def __init__(self, strategy: Strategy, agent: AgentOrchestrator, store: Store):
        self.strategy = strategy
        self.agent = agent
        self.store = store

    def generate(
        self, bars: Sequence[OHLCVBar], context: dict[str, str], portfolio: PortfolioSnapshot
    ) -> TradeSignal:
        output = self.strategy.generate(bars)
        proposed = self.agent.propose(output, context, portfolio)
        with self.store.transaction() as session:
            session.add_all(
                [
                    ObservationRecord(created_at=bar.timestamp, payload=bar.model_dump(mode="json"))
                    for bar in bars
                ]
            )
            session.add(
                SignalRecord(
                    created_at=proposed.timestamp, payload=proposed.model_dump(mode="json")
                )
            )
            session.add(
                AuditRecord(
                    created_at=proposed.timestamp,
                    payload={
                        "event": "signal_generated",
                        "signal": proposed.model_dump(mode="json"),
                    },
                )
            )
        return proposed
