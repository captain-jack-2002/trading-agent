from typing import Protocol

from trading_agent.models.domain import PortfolioSnapshot, TradeSignal


class AgentOrchestrator(Protocol):
    def propose(
        self, signal: TradeSignal, context: dict[str, str], portfolio: PortfolioSnapshot
    ) -> TradeSignal: ...


class ResearchAgent:
    """Deterministic context composition. No execution dependencies or LLM."""

    def propose(
        self, signal: TradeSignal, context: dict[str, str], portfolio: PortfolioSnapshot
    ) -> TradeSignal:
        return TradeSignal(
            symbol=signal.symbol,
            side=signal.side,
            timestamp=signal.timestamp,
            strategy=signal.strategy,
            explanation=(
                f"{signal.explanation}; "
                f"context source={context.get('source', 'unknown')}; equity INR {portfolio.equity}"
            ),
        )
