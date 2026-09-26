from typing import Protocol

from trading_agent.grounding import GroundingRequest, GroundingResult, GroundingService
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

    def ground(self, request: GroundingRequest, service: GroundingService) -> GroundingResult:
        """Return cited evidence or explicit abstention without generating numeric facts."""
        return service.query(request)

    def propose_grounded(
        self,
        signal: TradeSignal,
        portfolio: PortfolioSnapshot,
        request: GroundingRequest,
        service: GroundingService,
    ) -> TradeSignal:
        result = self.ground(request, service)
        if result.status == "abstained":
            return signal.model_copy(
                update={
                    "side": "hold",
                    "explanation": "insufficient_evidence: research proposal abstained",
                }
            )
        proposal = self.propose(signal, {"source": "typed_grounding"}, portfolio)
        return proposal.model_copy(
            update={
                "explanation": proposal.explanation
                + "; evidence="
                + ",".join(e.evidence_id for e in result.evidence)
            }
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
