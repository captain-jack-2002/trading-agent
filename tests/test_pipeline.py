from decimal import Decimal

from test_execution import broker  # noqa: F401
from test_risk import order
from test_strategy_agent import bars

from trading_agent.agent.orchestrator import ResearchAgent
from trading_agent.strategy.crossover import MovingAverageCrossover
from trading_agent.strategy.pipeline import ResearchPipeline


def test_agent_signal_persisted_and_risk_gated(broker):  # noqa: F811
    pipeline = ResearchPipeline(MovingAverageCrossover(2, 3), ResearchAgent(), broker.store)
    signal = pipeline.generate(
        bars([3, 2, 1, 4]), broker.provider.market_context(), broker.portfolio()
    )
    assert signal.side == "buy"
    # Order sizing remains explicit; an agent signal is not an executable approval.
    rejected = broker.submit(order(quantity=1000))
    assert rejected.status == "rejected"
    assert broker.portfolio().cash == Decimal("100000")
