from datetime import timedelta
from decimal import Decimal

import pytest
from test_risk import NOW, snapshot

from trading_agent.agent.orchestrator import ResearchAgent
from trading_agent.models.domain import Instrument, OHLCVBar
from trading_agent.strategy.crossover import MovingAverageCrossover


def bars(prices):
    return [
        OHLCVBar(
            instrument=Instrument(symbol="DEMO"),
            timestamp=NOW + timedelta(minutes=i),
            open=Decimal(p),
            high=Decimal(p),
            low=Decimal(p),
            close=Decimal(p),
            volume=10,
        )
        for i, p in enumerate(prices)
    ]


@pytest.mark.parametrize(
    ("prices", "side"), [([3, 2, 1, 4], "buy"), ([2, 3, 4, 1], "sell"), ([1, 2, 3, 4], "hold")]
)
def test_crossover(prices, side):
    signal = MovingAverageCrossover(fast=2, slow=3).generate(bars(prices))
    assert signal.side == side
    agent = ResearchAgent()
    assert agent.propose(signal, {"source": "mock"}, snapshot()).side == side
    assert not hasattr(agent, "broker")


def test_warmup_and_bad_bars():
    assert MovingAverageCrossover(2, 3).generate(bars([1])).side == "hold"
    with pytest.raises(ValueError):
        MovingAverageCrossover(3, 2)
    with pytest.raises(ValueError):
        MovingAverageCrossover(2, 3).generate(list(reversed(bars([1, 2, 3]))))


@pytest.mark.asyncio
async def test_agent_research_request_degrades_when_nse_is_unavailable():
    from contextlib import asynccontextmanager

    from trading_agent.integrations.nse_mcp import NSEMCPConfig, NSEMCPIntegration, NSEMCPProvider

    class OfflineSession:
        async def list_tools(self, **kwargs):
            raise ConnectionError("fixture offline")

        async def call_tool(self, name, arguments):
            raise AssertionError("must not call an undiscovered tool")

    @asynccontextmanager
    async def factory(url, timeout):
        yield OfflineSession()

    agent = ResearchAgent()
    provider = NSEMCPProvider(NSEMCPIntegration(NSEMCPConfig(), factory))
    context = await agent.request_nse_research(provider, "bhavcopy", "unknown", {})
    assert context.status == "research_context_unavailable"
