from datetime import datetime
from typing import Protocol

from trading_agent.models.domain import Instrument, OHLCVBar, Quote


class HistoricalDataProvider(Protocol):
    def historical(self, symbol: str, start: datetime, end: datetime) -> list[OHLCVBar]: ...


class MarketContextProvider(Protocol):
    def market_context(self) -> dict[str, str]: ...


class SymbolLookupProvider(Protocol):
    def lookup(self, symbol: str) -> Instrument | None: ...


class QuoteProvider(Protocol):
    def quote(self, symbol: str) -> Quote: ...


class NSEDataProvider(
    HistoricalDataProvider, MarketContextProvider, SymbolLookupProvider, QuoteProvider, Protocol
):
    """Boundary for future licensed/official market data; contains no transport assumptions."""
