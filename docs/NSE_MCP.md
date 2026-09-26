# Official NSE MCP research integration

The integration connects to the official NSE MCP servers over Streamable HTTP, using the official MCP Python SDK v2 (`mcp`). No authentication is currently documented by NSE.

| Provider | Endpoint |
| --- | --- |
| NSE Bhavcopy | `https://mcp.nseindia.in/bhavcopy/cm/mcp` |
| NSE CM Market | `https://mcp.nseindia.in/cmmkt/mcp` |

## Role and boundaries

NSE MCP is an optional informational and research context source. Tool names, descriptions, annotations, and input schemas are discovered from each server at runtime; no NSE tool names or response schemas are assumed. A normalized response is represented as `MCPResearchContext`, with source server, tool, UTC retrieval time, `informational_only=true`, `executable_price=false`, and `training_eligible=false` metadata. CM Market context additionally warns that information may be delayed and must not be treated as an executable quote.

MCP responses are not `HistoricalTrainingData` or `ExecutableMarketQuote` values. The MCP package has no broker or ML-pipeline dependency, and it does not persist raw responses. The optional Valkey cache is transient and TTL-bound. Valkey errors fall back to an uncached request. **MCP data must not be promoted into ML training datasets without separate lawful/licensed authorization.**

The intended data flow remains:

```text
Licensed NSE historical data
        ↓
ML training/backtesting

Official NSE MCP
        ↓
Agent research/context

Broker/licensed real-time data
        ↓
Executable price validation

Risk Engine
        ↓
HDFC SKY execution [future phase only]
```

The existing risk engine remains authoritative, and execution mode remains `paper`. NSE MCP clients only discover and request informational tool responses; they have no path to execution. The API exposes only read-only status and discovered tool metadata, with no HTTP tool-call route. Compose keeps published ports localhost-only and grants the app container outbound connectivity for NSE; database and Valkey remain on the private internal network.

## Configuration

The integration is enabled by default and can be disabled with `TRADING_NSE_MCP_ENABLED=false`. The endpoints can be overridden with `NSE_BHAVCOPY_MCP_URL` and `NSE_CM_MARKET_MCP_URL` (or their `TRADING_`-prefixed variants). Defaults are the official URLs above. `TRADING_NSE_MCP_TIMEOUT_SECONDS` (default `10`) bounds each operation, and `TRADING_NSE_MCP_CACHE_TTL_SECONDS` (default `60`) controls transient context caching. No credentials are required.

## Commands and API

- `trading-agent nse-mcp status` checks each configured server and reports availability.
- `trading-agent nse-mcp tools` lists dynamically discovered tool metadata grouped by server.
- `trading-agent nse-mcp test` checks initialization and tool discovery only. It never invokes a tool because generic metadata does not establish that an arbitrary operation is harmless.
- `GET /nse-mcp/status` and `GET /nse-mcp/tools` provide read-only status and discovery.
- `NSE_MCP_LIVE=1 ./scripts/check-nse-mcp.sh` runs the opt-in live initialization and discovery check against configured endpoints.

The agent orchestration can request a specifically selected tool through `NSEMCPProvider` and receives either typed context or `research_context_unavailable`. Generic invocation is available inside Python only; there is no generic unauthenticated HTTP tool execution route.

## Failure behavior and limits

Timeout, transport, protocol, malformed response, unknown tool, and schema validation failures are surfaced as integration errors or an unavailable context. They do not bypass risk checks, change the execution mode, or fabricate market data. Discovery results can change at any time. The SDK response is retained as structured JSON or content blocks without mapping it to licensed historical bars or executable quotes. CM Market timeliness and server-provided field semantics may vary; consumers must inspect discovered metadata and the response itself.

Normal tests use in-process fixtures and do not contact NSE. Live connectivity is opt-in and reports actual discoveries separately in `PHASE3_NSE_MCP_REPORT.md`.


## Phase 5A grounding boundary

`grounding.evidence_from_mcp` accepts available typed MCP context as ephemeral
informational evidence. It attaches no authoritative numeric observations, always
sets training/executable eligibility false, and applies freshness filtering.
Unavailable context returns no evidence. MCP text cannot satisfy executable-price,
volume/OI/derivative or other trading-critical fact requirements. Grounding and drift
commands do not discover/call NSE tools or introduce automatic network fallback.
