# Causal research features

`features.research.build_features(Sequence[ResearchBar])` produces immutable
Pydantic `FeatureRow` records. JSON uses ISO aware timestamps and explicit nulls.
The CLI adapts canonical Decimal bars to float research inputs with the complete
instrument identity (including expiry/strike/right), provenance, dataset version
and synthetic flag. Supply underlying price explicitly for basis/moneyness; there
is no automatic join that could introduce a future quote. Expiry is an explicit
aware instant, not an inferred exchange settlement time.

Each instrument must be strictly chronological; interleaving instruments is fine.
Duplicates and reversals fail. Features include simple/log returns at 1/5/10/20 bars,
20-bar SMA, seeded EMA12/26, Wilder RSI14 and ATR14, MACD/signal/histogram,
20-bar Bollinger bands and position, sample log-return realized volatility,
Parkinson high-low volatility, ATR/close, volume relative to the previous 20 bars,
volume z-score and second difference, price z-score and return acceleration,
MA distance, momentum, rolling highs/lows, raw OI and OI change, signed price-change
× OI-change relationship, basis, days to expiry, and underlying/strike moneyness.

Windows include the just-closed bar unless stated otherwise. Volatility is per bar,
not annualized. Bollinger uses sample standard deviation. EMAs seed from the first
observation; window features use null during warm-up. Constant-window z-scores,
zero prior volume, unavailable OI/underlying/expiry/strike remain null. No backward
filling occurs. `research-v1` fixes these definitions. Prefix-invariance tests verify
that appending future bars never changes prior feature rows. Research floats are
not accounting or order-price values.

An instrument history must keep the same provenance, dataset version and synthetic
status throughout; mixing these fails before feature calculation. This prevents
recursive/trailing features from silently carrying synthetic or different-source
history into a row claiming another lineage.
