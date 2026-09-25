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

## Phase 2 offline research

Phase 2 adds provider-mapped historical imports, immutable raw data, checksummed
manifests, canonical closed bars, explicit calendars, Parquet partitions, causal
features, purged walk-forward CPU baselines, configurable transaction costs and a
risk-gated backtester. It makes no network market-data requests and adds no live
broker connection. Read [the data pipeline](docs/DATA_PIPELINE.md),
[backtesting assumptions](docs/BACKTESTING.md), and [Phase 2 report](PHASE2_REPORT.md)
before using locally supplied licensed data.

Run the complete clearly labelled synthetic workflow with:

```bash
.venv/bin/python scripts/research_demo.py
```

It imports `examples/research/SYNTHETIC.csv`, engineers features, trains/evaluates a
CPU model, and compares SMA and model signals on the same held-out test window.
Outputs go under ignored `data/backtests/SYNTHETIC-<id>/`. These results verify
engineering only; they are not evidence of trading profitability.

For official or otherwise authorized data, first make an explicit provider column
mapping and import it with `.venv/bin/trading-agent data validate` / `data import`. Supply a
separately verified calendar, contract metadata, risk settings and dated cost rates
before running a backtest. Expected canonical fields and commands are in
[DATA_PIPELINE.md](docs/DATA_PIPELINE.md) and [PHASE2_REPORT.md](PHASE2_REPORT.md).


## Phase 4: SYNTHETIC model engineering

Phase 4 permits synthetic fixtures only. NSE MCP remains informational and never
training eligible or an executable price. All execution stays paper. These model
results are not evidence of expected market performance or live suitability.

Run the complete fixed experiment (all four targets, three CPU models, final
holdout, expanding/rolling folds, risk-gated baseline comparisons):

```bash
uv sync --frozen
uv run python scripts/phase4_training.py --report PHASE4_MODEL_TRAINING_REPORT.md
```

Generated datasets, models, predictions, metrics and checksums are stored under
ignored `data/models/PHASE4-SYNTHETIC-*/`. See the report for the exact run path.
The `--report` option writes/updates the human-readable report; omit it to retain
an existing report. The script uses only `examples/research/SYNTHETIC.csv`.

Individual commands extend the Phase 2 CLI:

```bash
uv run trading-agent features build data/datasets/<import-id>/manifest.json   --output data/features/direction.json --target direction --horizon 5
uv run trading-agent model train data/features/direction.json   --algorithm all --train-size 150 --validation-size 60 --test-size 85 --holdout
uv run trading-agent model list --registry data/models
uv run trading-agent model show <model-id> --registry data/models
uv run trading-agent model evaluate data/models/<model-id> data/features/direction.json   --output data/models/evaluation.json --trust-local-artifact
uv run trading-agent backtest run data/datasets/<import-id>/manifest.json   --calendar examples/research/calendar.json --config examples/research/backtest.json   --model data/models/<model-id> --trust-local-artifact --output data/backtests/model-run
```

For walk-forward training omit `--holdout`; add `--rolling` for fixed training
windows. Sizes/step count distinct timestamps before purging. `--holdout` requires
`--test-size` to equal the remaining timestamp groups. Folds purge label intervals
at partition and successive test-window boundaries. Final-holdout data must be
removed from walk-forward development input; the complete script does this.

Feature target choices: `direction`, `threshold` (`--threshold`, inclusive),
`barrier` (`--upper`, `--lower`), `future_return`. Regression uses `--algorithm ridge`,
`random_forest`, `lightgbm`, or `all`. No hyperparameter search is performed.
`backtest run --baseline cash|naive|sma` selects a fixed engineering baseline.
Models use probability >=0.5 or predicted return>0; regression can explicitly use
`--return-threshold`. `--help` documents every command. Serialized models execute
Python when loaded; only load your own trusted local artifacts. `model show`
checks metadata and SHA-256 without deserialization.

Existing feature artifacts without `source_bars` must be rebuilt before training.
New targets use `targets-v2` (inclusive threshold; unresolved barrier=negative,
ambiguous bar excluded). New causal distances/range use `research-v2`; old models
retain their original feature version and reject incompatible new features.
