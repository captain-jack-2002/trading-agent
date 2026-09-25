#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${NSE_MCP_LIVE:-}" != "1" ]]; then
  echo "Live NSE MCP checks are opt-in. Set NSE_MCP_LIVE=1 to continue." >&2
  exit 2
fi

exec uv run trading-agent nse-mcp test
