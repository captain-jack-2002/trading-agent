from typing import Protocol

from trading_agent.integrations.nse_mcp import MCPResearchContext, NSEMCPProvider, ServerSource
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

    async def request_nse_research(
        self,
        provider: NSEMCPProvider,
        source: ServerSource,
        tool_name: str,
        arguments: dict[str, object],
    ) -> MCPResearchContext:
        """Request an explicitly selected discovered tool as informational context."""
        return await provider.request_context(source, tool_name, arguments)
