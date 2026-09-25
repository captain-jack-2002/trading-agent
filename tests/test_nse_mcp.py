from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from mcp.types import CallToolResult, Tool

from trading_agent.config.settings import Settings
from trading_agent.integrations.nse_mcp import (
    MCPResearchContext,
    NSEMCPConfig,
    NSEMCPIntegration,
    NSEMCPProtocolError,
    NSEMCPProvider,
    NSEMCPResponseError,
    NSEMCPTimeoutError,
    UnknownToolError,
    config_from_settings,
)

TOOLS = [
    Tool(
        name="fixture_read_context",
        description="Read-only fixture tool",
        inputSchema={"type": "object", "properties": {"symbol": {"type": "string"}}},
        outputSchema={"type": "object"},
        annotations={"readOnlyHint": True},
    )
]


class FakeSession:
    def __init__(self, *, fail: Exception | None = None, delay: float = 0, response: Any = None):
        self.fail = fail
        self.delay = delay
        self.response = response
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self, **_: Any) -> Any:
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise self.fail
        return SimpleNamespace(tools=TOOLS, nextCursor=None)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((name, arguments))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise self.fail
        if self.response is not None:
            return self.response
        return CallToolResult(content=[], structuredContent={"observed": arguments})


def factory_for(session: FakeSession):
    @asynccontextmanager
    async def factory(url: str, timeout: float):
        assert url.startswith("https://")
        assert timeout > 0
        yield session

    return factory


def test_settings_use_official_defaults_and_can_disable() -> None:
    settings = Settings()
    config = config_from_settings(settings)
    assert config.bhavcopy_url == "https://mcp.nseindia.in/bhavcopy/cm/mcp"
    assert config.cm_market_url == "https://mcp.nseindia.in/cmmkt/mcp"
    assert config.enabled is True
    assert NSEMCPConfig(enabled=False).enabled is False


def test_settings_accept_documented_endpoint_environment_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NSE_BHAVCOPY_MCP_URL", "https://example.test/bhavcopy")
    monkeypatch.setenv("NSE_CM_MARKET_MCP_URL", "https://example.test/market")
    settings = Settings()
    assert settings.nse_bhavcopy_mcp_url == "https://example.test/bhavcopy"
    assert settings.nse_cm_market_mcp_url == "https://example.test/market"


@pytest.mark.asyncio
async def test_tool_discovery_and_invocation_metadata() -> None:
    session = FakeSession()
    integration = NSEMCPIntegration(NSEMCPConfig(), factory_for(session))
    tools = await integration.discover_bhavcopy_tools()
    assert [(tool.name, tool.server_source) for tool in tools] == [
        ("fixture_read_context", "bhavcopy")
    ]
    assert tools[0].input_schema["type"] == "object"
    assert tools[0].output_schema == {"type": "object"}
    result = await integration.call_bhavcopy_tool("fixture_read_context", {"symbol": "DEMO"})
    assert isinstance(result, MCPResearchContext)
    assert result.data == {"observed": {"symbol": "DEMO"}}
    assert result.source.source_server == "bhavcopy"
    assert result.source.tool_name == "fixture_read_context"
    assert result.source.informational_only is True
    assert result.source.executable_price is False
    assert result.source.training_eligible is False


@pytest.mark.asyncio
async def test_cm_market_context_is_marked_delayed_and_never_executable() -> None:
    integration = NSEMCPIntegration(NSEMCPConfig(), factory_for(FakeSession()))
    result = await integration.call_cm_market_tool("fixture_read_context", {})
    assert result.source.executable_price is False
    assert result.source.note
    assert "delayed" in result.source.note.lower()


@pytest.mark.asyncio
async def test_unknown_tool_is_rejected_before_invocation() -> None:
    session = FakeSession()
    integration = NSEMCPIntegration(NSEMCPConfig(), factory_for(session))
    with pytest.raises(UnknownToolError):
        await integration.call_bhavcopy_tool("missing", {})
    assert session.calls == []


@pytest.mark.asyncio
async def test_malformed_and_protocol_error_responses_are_rejected() -> None:
    malformed = NSEMCPIntegration(
        NSEMCPConfig(), factory_for(FakeSession(response=SimpleNamespace()))
    )
    with pytest.raises(NSEMCPResponseError):
        await malformed.call_bhavcopy_tool("fixture_read_context", {})

    protocol = NSEMCPIntegration(
        NSEMCPConfig(),
        factory_for(
            FakeSession(response=CallToolResult(content=[], structuredContent=None, isError=True))
        ),
    )
    with pytest.raises(NSEMCPProtocolError):
        await protocol.call_bhavcopy_tool("fixture_read_context", {})


@pytest.mark.asyncio
async def test_schema_mismatch_and_timeout_are_reported() -> None:
    schema_tool = Tool(
        name="fixture_requires_symbol",
        inputSchema={
            "type": "object",
            "required": ["symbol"],
            "properties": {"symbol": {"type": "string"}},
        },
    )

    class SchemaSession(FakeSession):
        async def list_tools(self, **_: Any) -> Any:
            return SimpleNamespace(tools=[schema_tool], nextCursor=None)

    integration = NSEMCPIntegration(NSEMCPConfig(), factory_for(SchemaSession()))
    with pytest.raises(ValueError, match="schema"):
        await integration.call_bhavcopy_tool("fixture_requires_symbol", {})

    timed = NSEMCPIntegration(
        NSEMCPConfig(timeout_seconds=0.01), factory_for(FakeSession(delay=0.1))
    )
    with pytest.raises(NSEMCPTimeoutError):
        await timed.discover_bhavcopy_tools()


@pytest.mark.asyncio
async def test_disabled_integration_returns_unavailable_provider_context() -> None:
    provider = NSEMCPProvider(NSEMCPIntegration(NSEMCPConfig(enabled=False)))
    result = await provider.request_context("bhavcopy", "fixture_read_context", {})
    assert result.status == "research_context_unavailable"
    assert result.data is None


@pytest.mark.asyncio
async def test_provider_converts_server_failure_to_unavailable() -> None:
    integration = NSEMCPIntegration(
        NSEMCPConfig(), factory_for(FakeSession(fail=ConnectionError("offline")))
    )
    result = await NSEMCPProvider(integration).request_context("cm_market", "any", {})
    assert result.status == "research_context_unavailable"
    assert result.source is None


@pytest.mark.asyncio
async def test_successful_context_is_cached_for_configured_ttl() -> None:
    class MemoryCache:
        value: str | None = None
        ttl: int | None = None

        def get(self, key: str) -> str | None:
            return self.value

        def set(self, key: str, value: str, ttl: int = 60) -> None:
            self.value = value
            self.ttl = ttl

    session = FakeSession()
    cache = MemoryCache()
    integration = NSEMCPIntegration(NSEMCPConfig(cache_ttl_seconds=17), factory_for(session))
    provider = NSEMCPProvider(integration, cache=cache)
    first = await provider.request_context("bhavcopy", "fixture_read_context", {})
    second = await provider.request_context("bhavcopy", "fixture_read_context", {})
    assert first.data == second.data
    assert first.source == second.source
    assert cache.ttl == 17
    assert len(session.calls) == 1


@pytest.mark.asyncio
async def test_cache_is_transient_and_cache_errors_degrade_gracefully() -> None:
    class FailingCache:
        def get(self, key: str) -> str | None:
            raise OSError("offline")

        def set(self, key: str, value: str, ttl: int = 60) -> None:
            raise OSError("offline")

    session = FakeSession()
    integration = NSEMCPIntegration(NSEMCPConfig(), factory_for(session))
    provider = NSEMCPProvider(integration, cache=FailingCache())
    first = await provider.request_context("bhavcopy", "fixture_read_context", {"symbol": "DEMO"})
    second = await provider.request_context("bhavcopy", "fixture_read_context", {"symbol": "DEMO"})
    assert first.data == second.data
    assert len(session.calls) == 2


def test_mcp_context_is_not_training_data_or_executable_quote() -> None:
    from trading_agent.models.domain import ExecutableMarketQuote, HistoricalTrainingData

    assert not issubclass(MCPResearchContext, HistoricalTrainingData)
    assert not issubclass(MCPResearchContext, ExecutableMarketQuote)


def test_mcp_integration_has_no_broker_or_training_dependencies() -> None:
    from pathlib import Path

    package = Path("src/trading_agent/integrations/nse_mcp")
    sources = "\n".join(path.read_text() for path in package.glob("*.py"))
    assert "trading_agent.execution" not in sources
    assert "trading_agent.ml" not in sources
