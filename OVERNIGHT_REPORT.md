# Phase 1 report — 2026-09-25

## Completed

- Python 3.12 / uv project with locked open-source dependencies and Apache-2.0 license.
- Typed environment settings and placeholder `.env.example`; no real `.env` created.
- Validated immutable domain models with Decimal money and aware timestamps.
- Historical/context/symbol/quote protocols and synthetic NSE mock; no scraping.
- Pure returns, SMA, EMA, sample rolling volatility, Wilder RSI/ATR, volume ratio.
- Deterministic crossover example and agent context orchestration; persisted signals.
- Risk-gated paper execution with all requested limits, cash/inventory checks,
  fresh quote validation, mandatory stop policy and durable duplicate protection.
- Buy/sell fills, weighted cost basis, cash, positions, realized/unrealized P&L.
- PostgreSQL schema, transactional ledger/audit persistence, restart recovery,
  serialized concurrent decisions and optional Valkey with TTL memory fallback.
- All eight requested API routes; no live-order route. HDFC stub always raises.
- Non-root container image, Compose app/PostgreSQL/Valkey, health checks and volumes.
- Architecture, risk, NSE, HDFC SKY, deployment and security documentation.
- Independent safety review completed; identified issues fixed and regression-tested.
- Logical foundation, application, and deployment/report commits; history preserved.

## Project structure

```text
src/trading_agent/
  api/          FastAPI endpoints and lifecycle
  config/       typed paper-only settings
  market_data/  provider protocols
  nse/          synthetic provider
  strategy/     interface, crossover and research pipeline
  features/     pure indicators
  models/       domain contracts (no ML dependency)
  agent/        orchestration interface and deterministic implementation
  risk/         policy evaluation
  execution/    abstract adapter, risk-gated paper adapter, disabled HDFC stub
  portfolio/    Decimal accounting
  persistence/  SQLAlchemy schema, transactions and recovery
  monitoring/   JSON logging and optional cache
 tests/         unit, API, concurrency, failure and opt-in service tests
 docs/          design, boundaries and operating instructions
 scripts/       checks, offline demo, isolated rootless service validation
```

## Verification evidence

| Check | Result |
|---|---|
| Full pytest suite with real PostgreSQL and Valkey sockets | **67 passed**, no skips or warnings |
| Source statement coverage in that run | **97%** (674 statements) |
| Ruff formatting and lint | Passed |
| Strict mypy over source | Passed, 33 source files |
| PostgreSQL integration | Concurrent duplicate submissions produce one fill; exposure limit and restart persistence verified |
| Valkey integration | Real connection and cache round-trip passed |
| `scripts/check-services.sh` | 2 real-service tests passed; temporary containers removed |
| Container image | Rootless Podman build passed |
| Container HTTP smoke | All read routes passed against PostgreSQL/Valkey; invalid-stop order rejected |
| Container runtime | Non-root, read-only root filesystem, dropped capabilities, `--network none` |
| Compose | `podman compose config` passed with installed provider |
| Package build | Wheel and source distribution built; artifact check confirms 33 Python modules and no local runtime files |
| Offline research demo | Buy proposal passed risk and filled; cash ₹99,000, exposure ₹1,000, equity ₹100,000 |

Without explicitly supplied local service socket paths, ordinary pytest runs skip
only the two opt-in service tests. `scripts/check-services.sh` provides those services.

Review regressions cover an advancing real-style clock, a provider returning
changing prices, and unsafe shared-connection in-memory SQLite. Quotes are now
fetched once per symbol, then evaluated with a clock captured after retrieval.
In-memory SQLite is rejected; file-backed SQLite remains available for local use.
A package-content regression check also catches empty wheels despite build success.

## Unresolved issues and scope limitations

- No unresolved lint, type, unit-test or service-test failures.
- Full `compose up` networking was not exercised. The file was validated, the image
  built, and app/PostgreSQL/Valkey were run using network-disabled socket-based
  containers. No host network settings were changed.
- Official NSE data, holidays/special sessions and HDFC SKY specifications are not
  supplied. All live functionality stays TODO and disabled.
- Paper fills omit fees, taxes, slippage, partial fills, leverage, shorts and exchange
  tick/lot rules. Stops validate intent but do not automatically trigger exits.
- Daily P&L uses last observed prior-day equity rather than an official closing mark.
  Risk blocks sells as well as buys at the daily loss limit; no bypass exists.
- API is unauthenticated and localhost-only. Passwordless database trust is for the
  isolated local sandbox. Production hardening and migrations remain later work.
- Container tags are not digest-pinned. Source/dependency versions are locked.
- uv emits a generic in-workspace-cache build warning; artifact inspection verifies
  that no cache, `.env`, database or container state is packaged.

## Workspace and safety notes

No sudo, live broker, real trade, broker credential, API key or host configuration
change was needed. Test containers used no network and workspace-local Podman
storage; all test containers were stopped and removed after verification.
Ignored tool/image caches and synthetic demo state remain in the workspace.

Early pytest runs used pytest's default temporary-path policy before explicit
workspace-local `--basetemp=.tmp/pytest` and `TMPDIR` were configured. Subsequent
verification uses workspace-local temporary files. No cleanup outside the workspace
was attempted. This is a deviation from the requested strict workspace confinement.

## Commands for the owner

No privileged setup or secrets are needed for the completed phase:

```bash
cd ~/Projects/trading-agent
uv sync --frozen
./scripts/check.sh
uv run python scripts/demo.py
./scripts/check-services.sh   # optional real services; existing rootless Podman required
podman compose up --build -d # optional persistent localhost sandbox
```

The demo has a fixed synthetic market-session clock and order ID; a second run
rejects that ID as a duplicate. The API uses wall time and rejects outside-session
orders. Do not change risk policy just to force a fill.

## External credentials and recommended next step

Phase 1 requires **no external credentials**. A later authorized NSE/licensed-data
integration may require entitlement credentials; HDFC SKY authentication requires
official specifications and a separately reviewed implementation. Do not provide
real credentials to this application.

Next: obtain official data specifications and calendar samples, add a licensed
historical-data adapter and reproducible offline backtests with realistic costs,
while retaining paper execution. Live integration should remain a separate phase.
