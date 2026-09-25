# Phase 3 NSE MCP implementation plan

## Goal and constraints

Add the official NSE Bhavcopy and CM Market MCP servers as optional, informational research sources. Their responses will use a domain type that cannot be passed as licensed historical training bars or executable quotes, will carry source and non-execution metadata, and will remain out of ML persistence. Existing risk gates and paper-only execution stay authoritative. Tool names and schemas come from runtime discovery.

## Steps

1. Add typed URL, enablement, timeout, and transient-cache TTL settings; add and lock the official MCP Python SDK v2 using `uv`.
2. Create `trading_agent.integrations.nse_mcp` with typed tool metadata and research context, separate Bhavcopy/CM Market providers, the SDK Streamable HTTP transport, generic discovery/invocation, response normalization, status, errors, structured logs, and an optional transient cache.
3. Add read-only CLI status/tools/test commands and API status/tools routes. Integrate optional MCP research at the agent context boundary with an unavailable fallback; do not connect this package to broker, risk, quote, or training-data interfaces.
4. Add offline fixture tests for settings, SDK lifecycle/discovery/calls, validation and failures, metadata, cache degradation, CLI/API behavior, and execution/training separation. Add an opt-in live discovery script that invokes no tool unless its schema gives a clearly harmless read operation.
5. Document endpoints, roles, boundaries, failure behavior, limitations, and the intended data flow. Create the phase report from actual validation results, including runtime tool discoveries when connectivity permits.
6. Run `./scripts/check.sh`, container build, and the offline MCP tests; resolve legitimate failures, then run the opt-in NSE check if reachable. Run Ruff, mypy, and `uv build`; commit logical changes only after all required gates pass.

## Validation boundary

Normal tests must be offline. Live requests are limited to MCP initialization, tool listing, and at most one discovered read-only tool. MCP responses are informational, potentially delayed, never executable, and never training-eligible. Execution remains `paper`; HDFC SKY stays unavailable.
