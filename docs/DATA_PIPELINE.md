# Local data pipeline

All examples in `examples/data/SYNTHETIC*` are **SYNTHETIC engineering fixtures**.
They demonstrate formats; they are neither historical market data nor evidence of
investment performance. No download, scraping, broker access, or live execution occurs.

## Canonical records

`trading_agent.data.schemas.canonical.CanonicalBar` is a frozen Pydantic record:

| Field | Contract |
| --- | --- |
| asset_class | equity, future, option |
| symbol, exchange | Explicit nonempty provider identifiers |
| timestamp | Aware **closed-bar availability timestamp**, normalized to UTC |
| open, high, low, close | Positive finite Decimal; low <= open/close <= high |
| volume | Nonnegative integer |
| underlying, expiry | Required for future and option; ISO date for expiry |
| strike, option_type | Required for options; positive Decimal and CE/PE |
| open_interest | Optional nonnegative integer for derivatives |
| contract | Optional embedded, immutable Contract metadata reference |

A source that timestamps bars at their opening must map/convert to an explicitly
known close time before import. The pipeline does not guess bar duration. Features
can use a complete record at its timestamp. Equivalent timezone offsets normalize
to the same UTC instant; derivative expiry validation uses Asia/Kolkata calendar
date (the Indian-market scope). No symbol parsing determines contract details.
Equities reject derivative metadata; futures reject option fields. Each derivative
identity includes underlying, expiry, strike and option side, so rollover contracts
remain distinct. Decimal strike formatting does not change instrument identity.

`Contract` requires contract_id, instrument_type (FUTSTK/FUTIDX/OPTSTK/OPTIDX),
symbol, exchange, underlying, expiry, explicitly supplied positive lot_size and
tick_size. Options also require strike and option_type. Supplied references must
match the bar identity; prices must align with the supplied tick. Data without a
reference remains usable for analysis; execution must require the applicable lot
and tick metadata. No current exchange lot sizes or expiry schedules are assumed.

## Provider mapping and import

`CSVMapping` JSON declares `provider`, `columns` (canonical name to CSV heading),
and optional `defaults`, `delimiter`, `encoding`, and `timestamp_format` (Python
strptime pattern, which must preserve timezone information). ISO timestamps are
accepted by default. Every required field must be mapped or explicitly defaulted.
There are no NSE column-name assumptions. Per-row derivative metadata can be
mapped; a full single-contract reference may be supplied in `defaults.contract`.

The CSV provider streams rows, rejects duplicate headers, missing mapped columns,
malformed widths, missing required values, inconsistent bars, duplicate timestamps
and non-increasing timestamps **per full instrument identity**. Interleaving distinct
instruments is permitted; callers requiring globally chronological data must sort
or merge explicitly. Errors identify the input row; failed imports publish nothing.

Run this exact local example from the repository root:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from trading_agent.data.providers import CSVMapping
from trading_agent.data.storage import import_csv, verify_import, load_parquet
mapping = CSVMapping.model_validate_json(Path('examples/data/equity_mapping.json').read_text())
manifest = import_csv(Path('examples/data/SYNTHETIC_equity.csv'), mapping,
                      Path('data/datasets'), synthetic=True)
folder = Path('data/datasets') / manifest.import_id
print(verify_import(folder).model_dump_json(indent=2))
print(sum(1 for _ in load_parquet(folder / 'parquet')))
PY
```

Imports are self-contained bundles at `data/datasets/<import_id>/`: `raw.csv`,
`mapping.json`, `normalized.jsonl`, `parquet/`, and `manifest.json`. This keeps
raw, normalized, and provenance files together for atomic publication. The input
bytes are copied before validation, then hashed; mapping, schema version, raw hash
and synthetic status define identity. An identical import raises FileExistsError;
there are no silent overwrites. A different mapping creates a different identity.
Synthetic provider names or filenames force the synthetic flag even if omitted.
Other inputs depend on the caller's truthful classification.

The manifest records provider, original filename, import time, schema version,
row count, UTC date range, full instrument identities, validation status, synthetic
flag, raw and mapping hashes, and SHA-256 of each artifact. `verify_import(folder)`
checks required artifacts, paths, all hashes, schema, identity and row statistics.
Checksums detect accidental corruption, not hostile rewriting of all local metadata.
Do not modify an imported bundle; import corrected data as a new version.

Normalized JSONL preserves Decimal strings. Parquet uses bounded batches (default
10,000, configurable 1..1,000,000) and `asset_class/year/month` UTC partitions, never
per-symbol partitions. Decimal values are exact base-10 strings with schema metadata,
rather than binary floating point. Contract metadata is JSON in a string column.
`load_parquet` reconstructs CanonicalBar objects in bounded batches; partition order
is not globally chronological. Small batch settings can create small files and are
intended for testing; choose an appropriate larger setting for large local datasets.
Chronology validation and manifest identities retain state proportional to instrument
count; price-row buffers remain bounded. No external market feed is bundled.

The equivalent CLI import (use a different root from an already run identical
example, or expect the intentional duplicate-import error) is:

```bash
.venv/bin/python -m trading_agent.cli data inspect examples/data/SYNTHETIC_equity.csv
.venv/bin/python -m trading_agent.cli data validate examples/data/SYNTHETIC_equity.csv --mapping examples/data/equity_mapping.json
.venv/bin/python -m trading_agent.cli data import examples/data/SYNTHETIC_equity.csv --mapping examples/data/equity_mapping.json --root data/datasets --synthetic
```

The printed `manifest_path` is the exact input for `data summarize`, feature builds,
and backtests. `data normalize` and `data build-parquet` use the same source/mapping
arguments and build the complete verified bundle; they do not mutate an existing
bundle or skip raw provenance.
