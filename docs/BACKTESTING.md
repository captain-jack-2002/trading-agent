# Offline backtesting

`BacktestEngine(settings, config, calendar).run(bars, strategy, contracts=None)`
returns `BacktestResult`. A strategy receives an immutable tuple of all **closed**
canonical bars and returns `SimulationOrder` values wrapping Phase 1
`OrderRequest`. `result.write_reports(directory)` writes `report.json` and
`report.md`; existing output is never overwritten. Supply explicit equity tick,
calendar, provenance, dataset version, strategy name/configuration and costs.
Default zero cost assumptions and synthetic labeling are for engineering only.

Canonical timestamps are close/availability times. Fixed duration `bar_seconds`
is supplied, never inferred. Open execution time is close minus duration.
Chronological groups may contain different instruments, but duplicate instrument
bars, backwards time, overlapping intervals and ambiguous symbol identities fail
closed. Signals generated at a close execute no earlier than the next matching
instrument bar open; equality between prior close and next open is allowed.
All same-time opens are marked before risk checks. Strategies cannot observe
that bar's high, low, close or volume at open. Limits execute only if marketable
at that open after adverse slippage and tick rounding; intrabar touch fills are
not simulated. Pending signals wait for their instrument's next bar; remaining
signals at dataset end are reported unfilled. Partial remainder is cancelled.

The supplied calendar must contain the complete bar span for trading and signal
generation. Its special sessions never override Phase 1 weekends, holidays or
clock restrictions. Every attempted full proposal passes the unchanged Phase 1
RiskEngine both at the raw open and adjusted execution price. Partial quantity is
also risk checked. Costs must fit available cash, including sell-side charges.
Duplicate IDs remain consumed even when rejected. Invalid strategy values are
rejected with audit reasons. Exceptions in strategy code terminate the run.

Full fills reject quantities exceeding the **previous closed bar's** volume.
Partial mode caps a configured fraction at that historical volume, rounding down
to contract lot size; zero remainder rejects. Reject mode explicitly rejects all
fills. These are deterministic research liquidity assumptions, not queue or order
book simulation. Use unique tradable symbols per contract. Derivative fills need
matching metadata, lot and tick reference data, explicit allowed instrument types
and cost product. All simulation is long-only and fully funded; it does not model
margin, marking-to-market settlement, option exercise, physical delivery or rolls.
No broker, credentials, network request, database or live executor is involved.

Stop-loss values are mandatory risk-eligibility intent under Phase 1 policy.
**The engine does not automatically trigger stop orders.** Overnight holdings and
intraday-labelled holdings are not forcibly closed; choose strategy exits and the
study period accordingly. Reported costs use the selected product, not inferred
holding periods. Daily loss baseline is the prior marked equity at day change,
before new opens, so overnight price gaps contribute to the day's risk P&L.

Accounting uses Decimal cash, weighted average entry price and proportional
allocation of entry fees on sells. Realized P&L is net of allocated entry and exit
fees. Unrealized P&L subtracts remaining entry fees. Equity equals cash plus marked
positions; equity minus initial capital reconciles realized plus unrealized P&L.
No artificial terminal liquidation is performed.

Metrics use the initial equity followed by each grouped close. `periods_per_year`
is an explicit sampling assumption (default 252 times 375 minute observations),
not a claim that irregular/missing intervals are regular. Total return is final /
initial minus one. Annualized compounded return and Calmar are null until at least
one configured year of observations exists. Volatility is sample standard
deviation of period returns times square-root periods/year. Sharpe uses mean
period return minus compounded period risk-free rate, annualized and divided by
annualized volatility. Sortino divides mean excess by root-mean-square negative
excess over **all** observations and multiplies by square-root periods/year.
Drawdown is greatest peak-to-trough fraction including starting capital. Calmar
is annualized return / maximum drawdown. Undefined denominators are null.

Each sell fill counts as one closed outcome (partial exits are separate outcomes).
Win/loss rates count positive/negative net outcomes over all sell fills; breakevens
remain in the denominator. Profit factor is gross positive outcome / absolute
negative outcome. Average loss is signed; expectancy is mean net outcome.
Turnover is both-sided execution notional / initial capital. Average exposure is
mean marked notional currency including initial zero. Fill count includes buys
and sells. Costs and slippage are separately reported currency amounts, with
slippage already embedded in fill prices. JSON contains exact trades, audit,
equity timestamps, assumptions, safe risk configuration and provenance. Database
and cache connection settings are excluded.

`compare_paper(simulated_equity, paper_equity, simulated_trade_count=...,
paper_trade_count=...)` compares caller-aligned recorded observations with no
account access, returning terminal/mean absolute equity and trade-count differences.
Timestamp alignment, dividends, corporate actions and comparable cost assumptions
remain the caller's responsibility. Synthetic examples demonstrate engineering
behavior only and provide no claim of profitability.

For fill reconciliation, normalize local records to `RecordedTrade` and call
`backtesting.comparison.compare_trades(simulated, paper)`. It matches unique
client-order IDs and reports quantity, fill-price and fee differences plus IDs
missing on either side. Differences are simulation minus paper. Explicitly
aggregate paper partial fills before calling; duplicate IDs are rejected to avoid
silently losing executions. Instrument/side identity and timestamp alignment
should be checked by the importing caller.

The historical-volume budget is shared by all fills at the same instrument/open.
Execution slippage changes acquisition basis and cash, while the market mark
remains the observed open until the next close. This avoids revaluing existing
holdings at a slipped transaction price during subsequent risk checks. Bankruptcy
at zero equity is reportable with total return -100% and drawdown 100%; return
ratios spanning a zero starting balance are undefined rather than dividing by
zero. Negative equity and externally recapitalized series require a different
cash-flow-aware accounting model and are rejected by this metrics API.
