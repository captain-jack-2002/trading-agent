from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal

from trading_agent.models.domain import Instrument, OHLCVBar, Quote


def utc_now() -> datetime:
    return datetime.now(UTC)


class MockNSEProvider:
    """Synthetic DEMO instrument only; never connects to NSE."""

    def __init__(self, clock: Callable[[], datetime] = utc_now):
        self.clock = clock
        self._prices = {"DEMO": Decimal("100")}
        self.bars: list[OHLCVBar] = []

    def set_price(self, symbol: str, price: Decimal) -> None:
        Quote(instrument=Instrument(symbol=symbol), price=price, timestamp=self.clock())
        self._prices[symbol] = price

    def lookup(self, symbol: str) -> Instrument | None:
        return Instrument(symbol=symbol) if symbol in self._prices else None

    def quote(self, symbol: str) -> Quote:
        instrument = self.lookup(symbol)
        if instrument is None:
            raise LookupError("symbol unavailable in mock data")
        return Quote(instrument=instrument, price=self._prices[symbol], timestamp=self.clock())

    def historical(self, symbol: str, start: datetime, end: datetime) -> list[OHLCVBar]:
        if start.tzinfo is None or end.tzinfo is None or start >= end:
            raise ValueError("valid aware time interval required")
        return [
            bar
            for bar in self.bars
            if bar.instrument.symbol == symbol and start <= bar.timestamp < end
        ]

    def market_context(self) -> dict[str, str]:
        return {"source": "mock", "description": "Synthetic engineering data; no market inference"}
