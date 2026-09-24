# NSE integration boundary

No website scraping or external market-data requests occur in Phase 1. No endpoint
URLs, payload schemas or licensing assumptions have been invented.

`market_data/interfaces.py` defines historical bars, market context, symbol lookup
and quotes as independent protocols, combined by `NSEDataProvider`. The mock
provider supplies synthetic `DEMO` quotes and injected historical bars. It labels
context and quotes as mock; it is not an NSE feed or a market-hours calendar.

Before adding an official/licensed provider, obtain the authorized specifications,
usage rights and symbol master. Map identifiers, exchange timestamps, instrument
types, intervals, corporate actions and adjusted/unadjusted prices explicitly.
Validate finite positive prices and chronological bars. Retain data source and
observation time; never replace a stale source timestamp with retrieval time.

TODO when official specifications are supplied: authenticated transport where
required, documented rate limits and retries, official holiday/special-session
calendar, symbol lifecycle handling, data-quality monitoring and contract tests
against approved sample payloads. Provider outage must fail closed for execution;
cached research data cannot silently become an executable quote.
