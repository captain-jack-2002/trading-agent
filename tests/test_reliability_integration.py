"""Fail-closed research/model integration; synthetic engineering data only."""

from datetime import UTC, datetime

from test_risk import snapshot
from test_strategy_agent import bars

from trading_agent.agent.orchestrator import ResearchAgent
from trading_agent.grounding import GroundingRequest, GroundingService
from trading_agent.strategy.crossover import MovingAverageCrossover


def test_grounded_agent_abstains_from_unsupported_proposal():
    agent = ResearchAgent()
    signal = MovingAverageCrossover(2, 3).generate(bars([3, 2, 1, 4]))
    result = agent.propose_grounded(
        signal,
        snapshot(),
        GroundingRequest(query="model limitations", as_of=datetime(2026, 1, 1, tzinfo=UTC)),
        GroundingService(),
    )
    assert result.side == "hold"
    assert "insufficient_evidence" in result.explanation


def test_backtest_model_health_can_reject_even_when_risk_allows(tmp_path):
    from decimal import Decimal

    from test_backtest_engine import bars as market_bars
    from test_backtest_engine import calendar
    from test_backtest_engine import order as sim_order

    from trading_agent.backtesting.engine import BacktestConfig, BacktestEngine, SimulationOrder
    from trading_agent.config.settings import Settings
    from trading_agent.ml.drift import ModelHealth

    health = ModelHealth(
        status="quarantined", diagnostics=(), window_count=20, threshold_version="engineering-test"
    )

    def strategy(history):
        return (
            [SimulationOrder(order=sim_order().order, model_health=health)]
            if len(history) == 1
            else []
        )

    result = BacktestEngine(
        Settings(), BacktestConfig(equity_tick_size=Decimal(".05")), calendar()
    ).run(market_bars(), strategy)
    assert not result.trades
    assert any("model_health_quarantined" in event.reasons for event in result.audit)


def test_grounded_research_proposal_retains_evidence_ids():
    from trading_agent.grounding import GroundingDocument, LocalGroundingIndex, SourceType

    signal = MovingAverageCrossover(2, 3).generate(bars([3, 2, 1, 4]))
    service = GroundingService(
        LocalGroundingIndex.from_documents(
            [
                GroundingDocument(
                    source_id="docs/limits.md",
                    source_type=SourceType.REPOSITORY_DOC,
                    content="Synthetic model limitations are engineering validation only.",
                )
            ]
        )
    )
    result = ResearchAgent().propose_grounded(
        signal,
        snapshot(),
        GroundingRequest(query="model limitations", as_of=datetime(2026, 1, 1, tzinfo=UTC)),
        service,
    )
    assert result.side == "buy"
    assert "; evidence=" in result.explanation
