# Phase 1 implementation plan

Build a paper-only modular Python application in the existing workspace. No live
network market integrations, broker credentials, or live execution will exist.

1. Establish typed settings and immutable domain models; add indicator and risk
   contract tests first. Use Decimal for money and timezone-aware timestamps.
2. Implement pure indicators, mock NSE interfaces, and deterministic crossover
   strategy. Keep reasoning independent of execution.
3. Implement a fail-closed risk engine and paper ledger. Serialize risk evaluation,
   fills, duplicate protection and persistence in one transaction. The adapter
   owns the risk gate: even a direct adapter call must undergo evaluation.
4. Add SQLAlchemy tables and ledger restoration, optional Valkey caching, JSON
   logging and FastAPI routes. Database failure must block orders. Keep one paper
   account and one process; reject multiple writers with a database ledger lock.
5. Add container configuration, deployment/integration/security documentation,
   run pytest, ruff, mypy and package build, fix failures and commit milestones.
6. Produce OVERNIGHT_REPORT.md with evidence and remaining integration work.

Assumptions: INR equities, integer quantities, market orders filled completely at
mock quote price, no fees/slippage/shorting/leverage. Stops are mandatory intent
metadata, not a simulation of an exchange-hosted protective order. Trading session
is configurable Asia/Kolkata weekdays with explicit holiday exclusions. Phase 1
is an engineering sandbox, not investment advice or a profitability claim.

Review focus: forged approvals/direct adapter calls, duplicate concurrent requests,
restart recovery, stale/mismatched/future quotes, day rollover and mark-to-market
loss, database errors, non-finite numbers, overselling and position cost basis.
