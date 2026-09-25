# Research transaction costs

No official tax, broker or exchange rates are embedded. The example configurations
are **SYNTHETIC, ILLUSTRATIVE ENGINEERING ASSUMPTIONS**, not historical or current
Indian charges. Supply a dated, independently verified schedule for the instrument,
segment, broker and study period before drawing economic conclusions.

`BacktestConfig.costs` maps `equity_delivery`, `equity_intraday`, `future`, and
`option` separately to `CostSchedule`. Omitted products fail closed. Each schedule
has brokerage, STT, exchange fees, SEBI fees and stamp duty. Each component's buy
and sell rates independently multiply execution price times executed quantity.
For options that base is premium turnover; futures use fully funded notional
turnover. Zero side rates represent non-applicable sides. Brokerage and any other
component can carry an optional per-fill absolute cap. GST multiplies only its
configured set of brokerage/exchange/SEBI amounts after caps. Duplicate taxable
component names are counted once. No rounding is silently applied to fees:
Decimal arithmetic preserves the supplied rates, and reports serialize exact
Decimal monetary values as strings. This is a configurable approximation, not a
broker contract-note calculator (no statutory minimums, exercise STT, delivery
depository charges, daily aggregation or exchange settlement).

`Slippage` supports zero, bps (`bps / 10000`), fractional percentage (`.01` means
1%) and an optional historical volatility multiplier. Volatility is the absolute
return between the last two closed bars of the instrument; it never observes the
execution bar close, range or volume. Components add. Buys move up and sells down.
Fill price rounds adversely to the explicitly supplied equity or contract tick.
A nonpositive adjusted price rejects the fill. Limits that this price would cross
are rejected, never silently filled beyond their limit. Slippage cost records
absolute difference from the next open times filled quantity, including rounding;
it is already reflected in portfolio cash and is not charged a second time.

Load JSON directly with `BacktestConfig.model_validate_json(path.read_text())`.
`examples/backtest/zero_costs.json` is a deliberately frictionless engineering
fixture. `illustrative_costs.json` demonstrates separate products and tax bases;
all numbers in that file are invented. Neither is suitable for investment claims.
