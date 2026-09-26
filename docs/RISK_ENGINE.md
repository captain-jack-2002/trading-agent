# Deterministic risk authority

Every paper order, including direct adapter calls, undergoes validation inside a
serialized ledger transaction. Callers provide intent only: ID, symbol, type,
side, positive integer quantity and optional stop. Price and portfolio are obtained
internally. The API accepts no risk decision, bypass switch or fill override.
A rejection never reaches the accounting transformation.

| Policy | Behavior |
|---|---|
| Capital per trade | Quantity × quote price must not exceed configured INR cap |
| Exposure | Buy must keep gross marked long exposure within INR cap |
| Open positions | New symbols cannot exceed configured count; additions do not count twice |
| Daily loss | Equity minus daily baseline <= negative limit blocks every order |
| Symbols/types | Explicit allowlists; paper accounting additionally supports only equities |
| Market hours | Asia/Kolkata weekday session, inclusive open and exclusive close; holidays excluded |
| Quotes | Identity/type match, positive finite price, aware time, age within limit; future quotes rejected |
| Portfolio marks | Every held symbol needs a fresh matching equity quote before approval |
| Duplicates | Persisted ID uniqueness, including rejected requests; survives process restart |
| Stop loss | Required by default; buy stop below quote, sell stop above quote |
| Cash/inventory | No leverage, negative cash, shorting or overselling |

Default limits: ₹10,000/order, ₹50,000 exposure, 5 positions, ₹5,000 daily loss,
30-second quote age, `DEMO` equity only. The default session is 09:15–15:30 IST
weekdays. These are engineering defaults, not recommended trading risk budgets.
A missing official holiday calendar is a known simulation limitation. Set
`TRADING_HOLIDAYS` to a JSON list of ISO dates. There is no hours bypass flag.

All relevant reasons are returned. Basic malformed requests fail Pydantic validation
before becoming an order; they are HTTP 422 and do not consume an ID. Valid-shaped
policy-rejected orders consume the ID and are audited. Use a new ID for revised
intent. Duplicate responses say rejected rather than replaying the original fill.

Daily P&L includes realized and unrealized changes. On an IST date change, the last
persisted equity becomes the baseline before new marks; overnight gaps count.
Without continuous observations this is a last-observed baseline, not an official
previous close. A limit breach blocks sells too; there is no emergency exemption.
The rule is evaluated against current equity, not a permanently latched daily halt.

Stops are metadata validation only; no automatic stop orders are simulated. Fees,
slippage, partial fills, tick/lot constraints, corporate actions and margin are
outside Phase 1. A caller with code/database modification privileges is outside
the API trust boundary; Python object privacy is not an OS security boundary.

Tests exercise boundaries, stale and future quotes, concurrent submissions,
restart duplicate recovery, accounting and transaction rollback.


## Phase 5A provenance and model health

Manual paper intent keeps the same risk path and existing audit event count;
`paper_order` now includes hashed decision provenance and typed fact decisions.
Model-backed paper requests use `ModelPaperExecutor`, which computes its own
prediction and fresh health and consults current lifecycle before submitting intent.
Quarantine abstains without creating an order. The registry lock spans the final
paper decision. The broker rejects untyped/free-form quote output and still obtains
its own ledger/quotes and calls the unchanged deterministic risk engine. A final
feature-timestamp check inside the ledger transaction also rejects inputs that
expired during inference/provider latency. A model score or grounding
confidence never authorizes a fill.
