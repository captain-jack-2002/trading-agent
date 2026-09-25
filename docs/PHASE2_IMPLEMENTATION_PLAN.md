# Phase 2 implementation plan

Build on `phase2-data-backtesting`; preserve Phase 1 source safety policies and
`v0.1.0-phase1` (`6b4ac627c9386bf249ae992809b794588fd3e2eb`). Execution stays paper.
No broker credentials, external market downloads, NSE scraping, host changes, or live orders.

## Design

Use a separate local research pipeline with typed immutable canonical bars,
explicit provider mappings and supplied calendars/contracts. Raw imports are
content-addressed, checksummed and never overwritten. Stream CSV validation and
bounded Parquet batches. Decimal prices remain exact at storage/accounting boundaries.

Backtests generate signals from closed bars and execute no earlier than the next
bar. Every proposed order passes the unchanged Phase 1 RiskEngine; simulated fill
prices are rechecked and costs must fit cash. Explicit calendars add restrictions,
never bypass Phase 1 policy. Derivatives data/features are supported; execution is
long-only fully funded research simulation, not exchange margin/settlement emulation.

Features use trailing information; targets explicitly record their future endpoint.
Chronological walk-forward splits purge overlapping label horizons. Fit transforms
on training only. CPU scikit-learn baselines and a checksum-verified local registry
are sufficient; no deep learning dependency. Reports retain data provenance and
prominent SYNTHETIC markings for engineering fixtures.

## Milestones

1. Data: canonical schemas, provider protocol/configured CSV, validation, immutable
   storage, Parquet, explicit calendar and contract reference models; tests/docs.
2. Research: causal features, labels, purged rolling/expanding splits, baseline
   training/evaluation and model registry; tests/docs.
3. Simulation: configurable Decimal costs, slippage, fills, risk-gated chronological
   backtester, accounting, metrics and JSON/Markdown reports; tests/docs.
4. Integration: CLI, labelled synthetic fixtures and end-to-end exercise; full
   regression suite, Ruff, strict mypy, package/container builds and coverage.
5. Review requirements/safety invariants; write PHASE2_REPORT.md with measured
   results, limitations, official input contracts, commands and Phase 3 suggestions.
   Commit stable logical milestones only after tests/lint/types pass.

Independent modules may be implemented concurrently with exclusive file ownership.
All writes and generated caches stay within this repository. Current branch is the
requested isolation boundary; do not create another branch or alter Phase 1 history.
