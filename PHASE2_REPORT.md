# Phase 2 report

## Status and safety

Phase 2 is implemented on `phase2-data-backtesting`. Execution remains `paper`.
Backtesting is a separate offline simulation: each order proposal passes the
existing Phase 1 `RiskEngine` at raw and adjusted open prices. It cannot construct
a live adapter or place broker orders. Phase 1 risk, execution, configuration,
API and persistence modules were not changed. The frozen tag
`v0.1.0-phase1` still resolves to `6b4ac627c9386bf249ae992809b794588fd3e2eb`.

All sample outcomes are marked **SYNTHETIC**. There are no market-performance
results in this report. Synthetic fixtures are used only to exercise parsing,
features, ML and execution mechanics.

## Architecture

The implementation provides frozen canonical equity/future/option bars and
FUTSTK/FUTIDX/OPTSTK/OPTIDX contract references; provider protocols and explicit
CSV column maps; row-level schema, ordering, duplicate and OHLC checks; content
addressed immutable imports with raw/mapping/artifact SHA-256 manifests; bounded
JSONL and `asset_class/year/month` Parquet writes; explicitly supplied sessions,
holidays, special sessions and expiry dates; instrument-local causal price, trend,
volatility, volume and optional F&O features; versioned forward labels; expanding
and rolling purged chronological splits; CPU Logistic Regression and Random
Forest baselines; a local checksum-verified model registry; Decimal cost and
slippage schedules; full/partial/reject and market/limit simulation; long-only
fully funded cash accounting behind the unchanged Phase 1 risk gate; metrics,
audit, JSON/Markdown output and local paper-comparison helpers.

Generated data belongs under ignored `data/` subdirectories. Each import is
published atomically as `data/datasets/<import_id>/` with raw, mapping, normalized,
Parquet and manifest files. Reports, features, datasets and model folders refuse
to overwrite existing artifacts.

## Modules and dependencies

Added `trading_agent.data` (schemas, providers, storage, calendar), causal research
features under `trading_agent.features`, `trading_agent.backtesting`,
`trading_agent.ml`, command integration in `trading_agent.cli`, and local synthetic
fixtures in `examples/data` and `examples/research`. New runtime libraries are
PyArrow (Parquet), NumPy, scikit-learn and joblib, all open source and CPU capable.
Python remains 3.12. There is no GPU requirement or deep-learning dependency.

## Data needed from you

No licensed exchange history, official calendar, or derivative contract master was
provided, so no official NSE integration or market backtest was performed. Before
an authorized import or economically meaningful run, provide:

- A locally available dataset you are entitled to use, plus its provider name,
  instrument universe, date range, timezone and bar timestamp convention.
- The corresponding explicit trading sessions, holidays and special sessions.
- Derivative contract master rows with actual contract ID/type, underlying,
  expiry, strike/right where applicable, lot size and tick size.
- Dated brokerage, STT, exchange, GST, SEBI and stamp assumptions for each product,
  verified against the broker/exchange documents applicable to that historical date.
- Any adjustment, corporate action, survivorship, roll, dividend and expiry policy
  appropriate to that dataset.

No undocumented NSE field, endpoint, entitlement or current statutory rate is
assumed. The CSV adapter consumes a mapping you supply; it does not claim that a
particular NSE download has a stable format.

## Expected input and future import commands

The source CSV must contain fields sufficient for a canonical record. Required
canonical fields are `asset_class`, `symbol`, `exchange`, timezone-aware closed-bar
`timestamp`, `open`, `high`, `low`, `close`, and nonnegative integer `volume`.
Futures/options also need `underlying` and ISO `expiry`; options need `strike` and
`option_type` (`CE`/`PE`). `open_interest` and an explicitly supplied nested `contract`
reference are optional when unavailable. Prices are positive Decimal-compatible
strings. Equities must not be given derivative fields. Per-instrument records must
be strictly time ordered; files may interleave distinct instruments.

There is deliberately no required NSE header spelling. Make a mapping JSON of this
form and replace each right-hand CSV header with the exact heading in your
licensed file. Add explicit defaults only when they are factually true for every
row in that file:

```json
{
  "provider": "<documented provider/export name>",
  "columns": {
    "asset_class": "<asset class heading>",
    "symbol": "<instrument symbol heading>",
    "exchange": "<exchange heading>",
    "timestamp": "<aware closed-bar timestamp heading>",
    "open": "<open heading>",
    "high": "<high heading>",
    "low": "<low heading>",
    "close": "<close heading>",
    "volume": "<volume heading>"
  },
  "defaults": {}
}
```

Validate, then import without overwriting an existing content identity:

```bash
.venv/bin/trading-agent data inspect /path/to/authorized.csv
.venv/bin/trading-agent data validate /path/to/authorized.csv --mapping provider-map.json
.venv/bin/trading-agent data import /path/to/authorized.csv --mapping provider-map.json --root data/datasets
.venv/bin/trading-agent data summarize data/datasets/<import_id>/manifest.json
```

The `manifest_path` printed by import is used by subsequent commands. `data
normalize` and `data build-parquet` currently run the same atomic complete import
pipeline so normalization cannot become detached from its raw source or manifest.
The equivalent Python API is in `docs/DATA_PIPELINE.md`. Timestamp input needs an
offset; do not map open-time timestamps as close-time timestamps without a known
source bar duration.

## Commands

Run all engineering workflows using the explicitly synthetic fixtures:

```bash
.venv/bin/python scripts/research_demo.py
```

The script imports data, builds features and five-bar labels, trains the logistic
baseline, evaluates its purged test fold, then runs SMA and ML backtests over the
same test timestamps. It prints artifact location and elapsed seconds. To use an
actual authorized import, first create a separate dated calendar and cost/risk JSON
matching `examples/research/backtest.json`; set `execution_mode` to `paper` and
include each symbol in the existing allowlist yourself. Then use:

```bash
.venv/bin/trading-agent features build data/datasets/<id>/manifest.json --output data/features/<name>.json --horizon 5 --target direction
.venv/bin/trading-agent model train data/features/<name>.json --registry data/models --algorithm logistic --train-size 1000 --validation-size 200 --test-size 200 --seed 42
.venv/bin/trading-agent model evaluate data/models/<model-id> data/features/<name>.json --output data/backtests/<name>-evaluation.json --trust-local-artifact
.venv/bin/trading-agent backtest run data/datasets/<id>/manifest.json --calendar calendar.json --config costs-risk-and-simulation.json --output data/backtests/<name>
```

Model loading uses `joblib` and can execute code. Only pass
`--trust-local-artifact` for a model artifact you created and trust. Use
`--start`/`--end` to select an inclusive closed-bar window; model backtests default
to the model's purged test fold and automatically retain prior bars as feature
warm-up, without trading during warm-up.

## Quality gates and measurements

- `./scripts/check.sh`: **148 passed, 2 skipped** (service tests use isolated
  sockets); **93% total statement coverage**.
- `ruff format --check src tests scripts` and `ruff check src tests scripts`:
  clean. `mypy src`: strict typing clean across 55 source modules.
- `uv build`: wheel and source distribution built; `scripts/check_artifacts.py`
  verified 55 Python modules in the wheel and no local runtime state.
- Rootless Podman image built as `localhost/trading-agent:phase2` (image ID
  `0ed24b99466b`). Running its installed CLI with `--help` succeeded offline.
- `./scripts/check-services.sh`: **2 passed**, isolated PostgreSQL and Valkey
  tests.
- Synthetic engineering benchmark: 100,000 streamed CSV rows imported and
  partitioned in 3.263 seconds; 10,000 causal feature rows built in 1.444 seconds;
  measured peak process RSS 148,404 KiB; 12 Parquet files. Measurements were made
  in one local CPython process on this workspace, include no database or model fit,
  and are not throughput guarantees. The feature builder materializes its input
  and output; the storage importer is chunked.
- Synthetic end-to-end workflow timing: 10.267 seconds for 300 bars, one
  logistic-regression walk-forward fold, test evaluation, and two same-window
  backtests (SMA and model). Local wall-clock observation only.

## Assumptions and limitations

No official calendar is bundled. An absent session is closed; the engine additionally
applies Phase 1's existing weekday, configured holiday, clock, quote, allowlist,
capital, stop-intent and position rules. Derivatives are modeled only as long,
fully funded holdings. Margin, daily settlement, exercise, delivery, rolls, automatic
stops, intrabar stop/limit touches and forced intraday liquidation are not modeled.
A stop-loss value meets the existing Phase 1 risk eligibility requirement; it does
not trigger execution. Full fills cap quantity against the previous closed bar's
volume; partial mode uses a configured fraction. There is no queue/order-book claim.

Costs have no embedded official/current rates. The zero and illustrative schedules
are research fixtures; GST bases and charge bases are explicit user configuration.
Slippage is an approximation. Equity tick size, derivative lot/tick and price
rounding reference must be supplied. Annualization follows configured sampling
intervals and remains undefined until at least a configured year of observations.

Parquet preserves Decimal fields as exact decimal strings, not Arrow Decimal types.
CSV validates chronology with state proportional to distinct contracts; Parquet
conversion buffers a configurable batch. Feature computation and ML datasets are
currently in memory. Backtesting retains chronological history to support arbitrary
strategies. For very large feature datasets, build an incremental feature/fit
interface before increasing beyond available memory. Bar adjustment quality,
corporate actions, survivorship and licensed-source completeness remain input
responsibilities. Pickle/joblib checksums detect corruption but do not establish a
publisher's trustworthiness.

## Recommended Phase 3

1. Adapt one official, licensed dataset only after documenting the supplied schema,
   entitlement, closed-bar timing and timestamp conventions; add a redacted fixture
   from user-supplied headers and validate raw checksums against source files.
2. Load official trading sessions, special sessions, expiries, lot sizes and tick
   sizes from user-provided versioned reference files; test calendar/contract version
   matching against datasets.
3. Add corporate-action aware adjusted data, instrument-universe/survivorship
   controls, futures/option settlement and margin models only from verified
   historical specifications.
4. Add batch/streaming causal feature computation and walk-forward report aggregation,
   then compare paper-ledger observations the user explicitly exports and supplies.
5. Keep all resulting execution paper-only; any live execution work would require a
   separate explicit phase, provider documentation and fresh safety review.
