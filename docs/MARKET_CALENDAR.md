# Explicit market calendars

`MarketCalendar` accepts explicit `sessions`, `holidays`, `expiries`, and `source`.
There is no weekday inference, public-holiday guessing, hardcoded expiry weekday,
or network lookup. An absent date is closed. A supplied Saturday special session
can open; an ordinary Monday without a supplied session remains closed.

Each `TradingSession` has `session_date`, aware `opens_at` / `closes_at`, and optional
`special: true`. Both endpoints must fall on the supplied local session date; open
must precede close. Duplicate dates, overlapping sessions, and holiday/session
conflicts are rejected. `session_for(aware_datetime)` returns the matching session
or None; `is_open` returns a boolean. Endpoints are inclusive to permit closed-bar
observations stamped at session close. A caller must separately ensure an execution
interval is valid; endpoint inclusion does not grant a new order after closing.
Naive timestamps are rejected. Date and time inputs must come from an authoritative
calendar maintained by the user; `expiries` are supplied reference dates, not inferred
contract expiry rules. Explicit restrictions do not bypass Phase 1 risk checks.

```bash
.venv/bin/python - <<'PY'
from datetime import datetime
from pathlib import Path
from trading_agent.data.calendar import MarketCalendar
calendar = MarketCalendar.model_validate_json(
    Path('examples/data/SYNTHETIC_calendar.json').read_text())
print(calendar.is_open(datetime.fromisoformat('2025-01-04T10:30:00+05:30')))
print(calendar.is_open(datetime.fromisoformat('2025-01-06T10:30:00+05:30')))
PY
```

The example prints True then False and is **SYNTHETIC**, not an official exchange
schedule. `SYNTHETIC_contracts.json` demonstrates all four derivative contract kinds;
its lot sizes, tick sizes and expiry dates are fictional. No current NSE contract
specifications are asserted by these fixtures.
