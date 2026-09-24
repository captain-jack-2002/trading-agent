from collections.abc import Sequence
from typing import Protocol

from trading_agent.models.domain import OHLCVBar, TradeSignal


class Strategy(Protocol):
    def generate(self, bars: Sequence[OHLCVBar]) -> TradeSignal: ...
