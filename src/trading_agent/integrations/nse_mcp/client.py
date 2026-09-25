"""Runtime discovery and informational tool invocation for official NSE MCP servers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import Any, Literal, Protocol

from jsonschema import ValidationError, validate
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError
from pydantic import BaseModel, ConfigDict

from trading_agent.config.settings import Settings

logger = logging.getLogger(__name__)
ServerSource = Literal["bhavcopy", "cm_market"]
ContextStatus = Literal["available", "research_context_unavailable"]

BHAVCOPY_URL = "https://mcp.nseindia.in/bhavcopy/cm/mcp"
CM_MARKET_URL = "https://mcp.nseindia.in/cmmkt/mcp"
CM_MARKET_NOTE = (
    "CM Market information may be delayed and must not be treated as an executable quote."
)


class NSEMCPError(RuntimeError):
    """Base class for errors at the informational NSE MCP boundary."""


class IntegrationDisabledError(NSEMCPError):
    pass


class NSEMCPTimeoutError(NSEMCPError):
    pass


class NSEMCPProtocolError(NSEMCPError):
    pass


class NSEMCPResponseError(NSEMCPError):
    pass


class UnknownToolError(NSEMCPError):
    pass


class ToolSchemaError(ValueError):
    pass


@dataclass(frozen=True)
class NSEMCPConfig:
    enabled: bool = True
    bhavcopy_url: str = BHAVCOPY_URL
    cm_market_url: str = CM_MARKET_URL
    timeout_seconds: float = 10.0
    cache_ttl_seconds: int = 60

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.cache_ttl_seconds <= 0:
            raise ValueError("cache_ttl_seconds must be positive")
        for url in (self.bhavcopy_url, self.cm_market_url):
            if not url.startswith("https://"):
                raise ValueError("NSE MCP endpoints must use HTTPS")


def config_from_settings(settings: Settings) -> NSEMCPConfig:
    return NSEMCPConfig(
        enabled=settings.nse_mcp_enabled,
        bhavcopy_url=settings.nse_bhavcopy_mcp_url,
        cm_market_url=settings.nse_cm_market_mcp_url,
        timeout_seconds=settings.nse_mcp_timeout_seconds,
        cache_ttl_seconds=settings.nse_mcp_cache_ttl_seconds,
    )


class NSEMCPTool(BaseModel):
    """Typed runtime metadata copied from the server's discovered tool definition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str | None
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None = None
    server_source: ServerSource
    read_only_hint: bool = False
    required_arguments: tuple[str, ...] = ()


class ResearchSourceMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_server: ServerSource
    tool_name: str
    retrieval_timestamp: datetime
    informational_only: Literal[True] = True
    executable_price: Literal[False] = False
    training_eligible: Literal[False] = False
    note: str | None = None


class MCPResearchContext(BaseModel):
    """Ephemeral server response; deliberately distinct from quote and training types."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: ContextStatus
    data: Any = None
    source: ResearchSourceMetadata | None = None
    reason: str | None = None


class MCPClientProtocol(Protocol):
    async def list_tools(self, *, cursor: str | None = None) -> Any: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...


SessionFactory = Callable[[str, float], AbstractAsyncContextManager[MCPClientProtocol]]


@asynccontextmanager
async def _sdk_session(url: str, timeout_seconds: float) -> AsyncIterator[MCPClientProtocol]:
    """The official SDK selects Streamable HTTP from a URL endpoint."""
    transport = streamable_http_client(url, terminate_on_close=False)
    async with Client(
        transport, raise_exceptions=True, read_timeout_seconds=timeout_seconds
    ) as client:
        yield client


class _ServerClient:
    def __init__(
        self,
        source: ServerSource,
        url: str,
        config: NSEMCPConfig,
        session_factory: SessionFactory,
    ) -> None:
        self.source = source
        self.url = url
        self.config = config
        self.session_factory = session_factory

    def _require_enabled(self) -> None:
        if not self.config.enabled:
            raise IntegrationDisabledError("NSE MCP integration is disabled")

    async def discover_tools(self) -> list[NSEMCPTool]:
        self._require_enabled()
        started = perf_counter()
        try:
            async with asyncio.timeout(self.config.timeout_seconds):
                async with self.session_factory(self.url, self.config.timeout_seconds) as client:
                    tools: list[NSEMCPTool] = []
                    cursor: str | None = None
                    while True:
                        result = await client.list_tools(cursor=cursor)
                        page = getattr(result, "tools", None)
                        if not isinstance(page, list):
                            raise NSEMCPResponseError("MCP list_tools response has no tools list")
                        for tool in page:
                            name = getattr(tool, "name", None)
                            schema = getattr(tool, "input_schema", None)
                            if not isinstance(name, str) or not isinstance(schema, dict):
                                raise NSEMCPResponseError("MCP tool definition is malformed")
                            annotations = getattr(tool, "annotations", None)
                            if isinstance(annotations, BaseModel):
                                annotation_data = annotations.model_dump(
                                    mode="json", exclude_none=True
                                )
                            elif isinstance(annotations, dict):
                                annotation_data = annotations
                            else:
                                annotation_data = {}
                            required = schema.get("required", [])
                            tools.append(
                                NSEMCPTool(
                                    name=name,
                                    description=getattr(tool, "description", None),
                                    input_schema=schema,
                                    output_schema=getattr(tool, "output_schema", None),
                                    server_source=self.source,
                                    read_only_hint=annotation_data.get("readOnlyHint") is True,
                                    required_arguments=tuple(
                                        item for item in required if isinstance(item, str)
                                    ),
                                )
                            )
                        cursor = getattr(result, "next_cursor", None)
                        if not cursor:
                            break
            logger.info(
                "nse_mcp_tool_discovery",
                extra={
                    "source_server": self.source,
                    "tool_count": len(tools),
                    "tool_names": [tool.name for tool in tools],
                    "latency_ms": round((perf_counter() - started) * 1000, 2),
                    "success": True,
                },
            )
            logger.info(
                "nse_mcp_connection",
                extra={"source_server": self.source, "success": True},
            )
            return tools
        except TimeoutError as exc:
            self._log_failure("tool_discovery", started, "timeout")
            raise NSEMCPTimeoutError(f"{self.source} MCP discovery timed out") from exc
        except NSEMCPError:
            self._log_failure("tool_discovery", started, "protocol_or_response_error")
            raise
        except (MCPError, OSError, ConnectionError) as exc:
            self._log_failure("tool_discovery", started, "connection_or_protocol_error")
            raise NSEMCPProtocolError(f"{self.source} MCP server unavailable") from exc
        except Exception as exc:
            self._log_failure("tool_discovery", started, "protocol_error")
            raise NSEMCPProtocolError(f"{self.source} MCP discovery failed") from exc

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> MCPResearchContext:
        self._require_enabled()
        started = perf_counter()
        try:
            tools = await self.discover_tools()
            selected = next((tool for tool in tools if tool.name == name), None)
            if selected is None:
                raise UnknownToolError(f"Tool {name!r} was not discovered on {self.source}")
            try:
                validate(instance=arguments, schema=selected.input_schema)
            except ValidationError as exc:
                raise ToolSchemaError(
                    f"arguments do not match the discovered tool schema: {exc.message}"
                ) from exc
            async with asyncio.timeout(self.config.timeout_seconds):
                async with self.session_factory(self.url, self.config.timeout_seconds) as client:
                    result = await client.call_tool(name, arguments)
            if not hasattr(result, "is_error") or not hasattr(result, "content"):
                raise NSEMCPResponseError("MCP tool response is malformed")
            if result.is_error:
                raise NSEMCPProtocolError(f"{self.source} MCP tool returned an error")
            structured = getattr(result, "structured_content", None)
            content = getattr(result, "content", None)
            if structured is not None:
                data = _json_value(structured)
            elif isinstance(content, list) and content:
                data = [_json_value(item) for item in content]
            else:
                raise NSEMCPResponseError("MCP tool response contains no data")
            context = MCPResearchContext(
                status="available",
                data=data,
                source=ResearchSourceMetadata(
                    source_server=self.source,
                    tool_name=name,
                    retrieval_timestamp=datetime.now(UTC),
                    note=CM_MARKET_NOTE if self.source == "cm_market" else None,
                ),
            )
            logger.info(
                "nse_mcp_tool_invocation",
                extra={
                    "source_server": self.source,
                    "tool_name": name,
                    "latency_ms": round((perf_counter() - started) * 1000, 2),
                    "success": True,
                },
            )
            return context
        except TimeoutError as exc:
            self._log_failure("tool_invocation", started, "timeout", name)
            raise NSEMCPTimeoutError(f"{self.source} MCP tool call timed out") from exc
        except (NSEMCPError, ToolSchemaError):
            self._log_failure("tool_invocation", started, "request_failed", name)
            raise
        except (MCPError, OSError, ConnectionError) as exc:
            self._log_failure("tool_invocation", started, "connection_or_protocol_error", name)
            raise NSEMCPProtocolError(f"{self.source} MCP server unavailable") from exc
        except Exception as exc:
            self._log_failure("tool_invocation", started, "protocol_error", name)
            raise NSEMCPProtocolError(f"{self.source} MCP tool invocation failed") from exc

    def _log_failure(
        self, operation: str, started: float, error: str, name: str | None = None
    ) -> None:
        logger.warning(
            "nse_mcp_operation_failed",
            extra={
                "source_server": self.source,
                "operation": operation,
                "tool_name": name,
                "latency_ms": round((perf_counter() - started) * 1000, 2),
                "error_class": error,
                "success": False,
            },
        )


def _json_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True, exclude_none=True)
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise NSEMCPResponseError("MCP response cannot be represented as JSON") from exc
    return value


class NSEMCPIntegration:
    def __init__(self, config: NSEMCPConfig, session_factory: SessionFactory = _sdk_session):
        self.config = config
        self.bhavcopy = _ServerClient("bhavcopy", config.bhavcopy_url, config, session_factory)
        self.cm_market = _ServerClient("cm_market", config.cm_market_url, config, session_factory)

    async def discover_bhavcopy_tools(self) -> list[NSEMCPTool]:
        return await self.bhavcopy.discover_tools()

    async def discover_cm_market_tools(self) -> list[NSEMCPTool]:
        return await self.cm_market.discover_tools()

    async def call_bhavcopy_tool(self, name: str, arguments: dict[str, Any]) -> MCPResearchContext:
        return await self.bhavcopy.call_tool(name, arguments)

    async def call_cm_market_tool(self, name: str, arguments: dict[str, Any]) -> MCPResearchContext:
        return await self.cm_market.call_tool(name, arguments)

    async def status(self) -> dict[str, Any]:
        async def inspect(source: ServerSource, client: _ServerClient) -> dict[str, Any]:
            if not self.config.enabled:
                return {"source_server": source, "status": "disabled", "tool_count": 0}
            started = perf_counter()
            try:
                tools = await client.discover_tools()
                return {
                    "source_server": source,
                    "status": "available",
                    "tool_count": len(tools),
                    "latency_ms": round((perf_counter() - started) * 1000, 2),
                }
            except NSEMCPError as exc:
                return {
                    "source_server": source,
                    "status": "unavailable",
                    "tool_count": 0,
                    "error": str(exc),
                }

        bhavcopy, cm_market = await asyncio.gather(
            inspect("bhavcopy", self.bhavcopy), inspect("cm_market", self.cm_market)
        )
        return {"enabled": self.config.enabled, "servers": [bhavcopy, cm_market]}


class CacheProtocol(Protocol):
    def get(self, key: str) -> str | None: ...

    def set(self, key: str, value: str, ttl: int = 60) -> None: ...


class NSEMCPProvider:
    """Context facade used by orchestration; it has no quote, training, or broker methods."""

    def __init__(self, integration: NSEMCPIntegration, cache: CacheProtocol | None = None):
        self.integration = integration
        self.cache = cache

    async def request_context(
        self, source: ServerSource, tool_name: str, arguments: dict[str, Any]
    ) -> MCPResearchContext:
        if source not in ("bhavcopy", "cm_market"):
            raise ValueError("source must identify an NSE MCP server")
        client = self.integration.bhavcopy if source == "bhavcopy" else self.integration.cm_market
        method = (
            self.integration.call_bhavcopy_tool
            if source == "bhavcopy"
            else self.integration.call_cm_market_tool
        )
        cache_key = (
            "nse-mcp:"
            + hashlib.sha256(
                json.dumps(
                    [source, tool_name, arguments], sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
        )
        if self.cache is not None:
            try:
                cached = self.cache.get(cache_key)
                if cached is not None:
                    logger.info("nse_mcp_cache", extra={"source_server": source, "cache": "hit"})
                    return MCPResearchContext.model_validate_json(cached)
            except Exception:
                logger.warning(
                    "nse_mcp_cache", extra={"source_server": source, "cache": "read_failure"}
                )
            logger.info("nse_mcp_cache", extra={"source_server": source, "cache": "miss"})
        try:
            context = await method(tool_name, arguments)
        except (NSEMCPError, ValueError, TypeError) as exc:
            return MCPResearchContext(
                status="research_context_unavailable",
                reason=type(exc).__name__,
            )
        if self.cache is not None:
            try:
                self.cache.set(
                    cache_key,
                    context.model_dump_json(),
                    ttl=client.config.cache_ttl_seconds,
                )
            except Exception:
                logger.warning(
                    "nse_mcp_cache", extra={"source_server": source, "cache": "write_failure"}
                )
        return context
