# Trading Agent — Phase 1

An open-source Python 3.12 foundation for AI-assisted Indian-market research.
**Paper execution only.** No live brokerage integration, credentials, scraping,
LLM dependency or profitability claims. The default symbol `DEMO` and its quotes
are synthetic; changing the allowlist does not provide real market data.

## Run locally

Install Python 3.12 and uv using your existing user-level tooling, then:

```bash
uv sync --frozen
uv run python scripts/demo.py
./scripts/check.sh
```

The demo uses a fixed in-session clock and a workspace SQLite database. Repeating
it reuses the same order ID and demonstrates duplicate rejection. SQLite is only
a development/test option; the deployed application uses PostgreSQL.

Start the stack with an already configured rootless Podman and Compose provider:

```bash
podman compose up --build -d
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
```

`docker compose` can use the same file. No `.env` or credentials are required for
this isolated local sandbox. PostgreSQL and Valkey have no published ports.
The internal container network has no external egress. Do not expose this
unauthenticated sandbox remotely.

For a dependency-free local API session after `uv sync`:

```bash
TRADING_DATABASE_URL=sqlite:///paper.db uv run uvicorn trading_agent.api.app:app --host 127.0.0.1
```

Paper orders honor the configured market session and real wall clock:

```bash
curl -X POST http://127.0.0.1:8000/paper/orders \
  -H 'Content-Type: application/json' \
  -d '{"client_order_id":"example-1","symbol":"DEMO","side":"buy","quantity":10,"stop_loss":"90"}'
```

An outside-session request receives an audited rejection. Never disable checks to
make a demo fill; use `scripts/demo.py` with its explicit synthetic clock.

## Interfaces

| Route | Purpose |
|---|---|
| `GET /health` | Process liveness and paper mode |
| `GET /ready` | Required ledger availability; optional cache status |
| `GET /portfolio` | Current marked cash, equity and P&L |
| `GET /positions` | Open long positions |
| `GET /signals` | Persisted research signals (initially empty) |
| `GET /risk/status` | Policy limits and daily loss state |
| `POST /paper/orders` | Risk-gated simulated fill or rejection |
| `GET /audit` | Most recent audit events |

List routes accept a bounded `limit` (1–1000). Order rejection is HTTP 422;
unavailable required infrastructure is 503. No live-order route exists. API docs
are at `/docs`. `ResearchPipeline` and the demo show how to generate signals.

## Design and limitations

Agent proposals pass through deterministic risk evaluation before any fill.
The paper adapter itself owns that evaluation: a `RiskDecision` cannot be supplied
as an authorization. Monetary values use Decimal and serialize as strings.

The simulation is long-only equities, whole shares, instant full fills at the
provider quote, with no fees, slippage or exchange matching. Stop loss is validated
intent metadata; there is no automatic stop execution yet. Daily P&L uses the last
observed prior-day equity as its baseline, not an official exchange close. Holiday
and special-session calendars require official data before realistic use.

See [architecture](docs/ARCHITECTURE.md), [risk](docs/RISK_ENGINE.md),
[NSE boundary](docs/NSE_INTEGRATION.md), [HDFC SKY boundary](docs/HDFC_SKY_INTEGRATION.md),
[deployment](docs/DEPLOYMENT.md), [security](docs/SECURITY.md), and
[implementation report](OVERNIGHT_REPORT.md).

License: Apache-2.0. Runtime and development dependencies are open-source.
