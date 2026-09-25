# Phase 3 NSE MCP integration report

## SDK and endpoints

- Official Model Context Protocol Python SDK: `mcp==2.2.0` (locked by `uv.lock`), Streamable HTTP transport.
- NSE Bhavcopy: `https://mcp.nseindia.in/bhavcopy/cm/mcp`
- NSE CM Market: `https://mcp.nseindia.in/cmmkt/mcp`
- No authentication is configured or required by this integration.

## Runtime discovery and connectivity

On 2026-09-25, both endpoints initialized and returned tool lists. Discovery found the following names; all descriptions, input schemas, optional output schemas, and annotations are read from the server at runtime.

Bhavcopy tools (13):

`get_top_by_volume`, `get_top_movers`, `get_volume_analysis`, `nse_lookup_symbol`, `get_stock_history`, `search_symbols`, `get_market_breadth`, `get_corporate_actions`, `compare_stocks`, `get_ltp_by_date`, `get_bulk_quote`, `moving_average`, `get_52_week_high_low`.

CM Market tools (13):

`cm_get_sme_stocks`, `cm_get_live_market_data`, `cm_get_equity_stocks`, `nse_get_losers`, `cm_get_call_auction_stocks`, `cm_get_bond_stocks`, `cm_get_live_gainers`, `nse_get_gainers`, `cm_get_data_status`, `cm_get_live_losers`, `cm_get_stock_quote`, `nse_get_market_movers`, `cm_get_allstocks_status`.

The successful status probe measured 574.65 ms for Bhavcopy discovery and 611.49 ms for CM Market discovery. A later live check timed out on both servers at the configured 10-second operation limit. Availability is therefore intermittent; the integration reports unavailable context and continues without NSE data. The successful tool discovery showed no explicit read-only annotations, so the live check intentionally invoked no tools.

No tool responses were requested. Response schema variation could not be assessed from tool results; discovered input schemas vary by operation, and any advertised output schemas are retained as runtime metadata when provided.

## Changes

- Added `trading_agent.integrations.nse_mcp` with separate server clients, dynamic discovery, JSON Schema argument validation, generic invocation, response normalization, typed research metadata, timeout/protocol errors, health status, structured logs, and transient optional caching.
- Added typed `MCPResearchContext`, `HistoricalTrainingData`, and `ExecutableMarketQuote` boundaries. CM Market context carries an explicit delayed/informational warning.
- Added typed settings, `nse-mcp status|tools|test` CLI commands, read-only `GET /nse-mcp/status` and `GET /nse-mcp/tools`, and an opt-in `scripts/check-nse-mcp.sh`.
- Exposed the optional research provider to orchestration through `ResearchAgent.request_nse_research`. Failures return `research_context_unavailable`.
- Added `docs/NSE_MCP.md`. Compose keeps the app port bound to localhost, attaches only the app to outbound NSE connectivity, and leaves PostgreSQL and Valkey on their isolated network.

## Validation

- `./scripts/check.sh`: passed. Ruff formatting/lint, mypy, offline tests, `uv build`, and the artifact check all passed.
- Offline suite: 163 passed, 2 skipped because optional local PostgreSQL and Valkey test sockets were not configured.
- Targeted offline MCP, API, CLI, and agent tests: passed.
- `podman build -t trading-agent:phase3-official-nse-data .`: passed.
- Opt-in live check: both endpoints succeeded during one discovery run; a later check timed out on both. No tools were invoked.

## Safety and limitations

MCP output is represented only as informational research context with `informational_only=true`, `executable_price=false`, and `training_eligible=false`. Raw responses are not written to ML datasets; the optional cache is transient and TTL-bound. No MCP package path imports brokerage or ML execution code, and MCP-derived context does not become a licensed historical bar or executable quote. MCP cannot directly place orders. `execution_mode` remains `paper`; HDFC SKY live execution remains unavailable.

CM Market prices may be delayed and are not executable prices. Tool lists and schemas can change, availability was intermittent during live checks, and response field semantics are server-defined. **MCP data must not be promoted into ML training datasets without separate lawful/licensed authorization.**

Recommended next step: monitor server availability and schema evolution through the discovery/status endpoints before selecting any discovered research tool for routine agent use.
