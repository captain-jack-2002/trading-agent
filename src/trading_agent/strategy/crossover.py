from collections.abc import Sequence
from typing import Literal

from trading_agent.models.domain import OHLCVBar, TradeSignal


class MovingAverageCrossover:
    """Engineering example only. Requires closed bars in increasing time order."""

    def __init__(self, fast: int = 5, slow: int = 20):
        if not 0 < fast < slow:
            raise ValueError("require 0 < fast < slow")
        self.fast, self.slow = fast, slow

    def generate(self, bars: Sequence[OHLCVBar]) -> TradeSignal:
        if not bars:
            raise ValueError("at least one bar required")
        if any(bar.instrument != bars[0].instrument for bar in bars):
            raise ValueError("mixed instruments")
        if any(a.timestamp >= b.timestamp for a, b in zip(bars, bars[1:], strict=False)):
            raise ValueError("bars must be strictly increasing")
        side: Literal["buy", "sell", "hold"] = "hold"
        explanation = "Warm-up: insufficient closed bars"
        if len(bars) > self.slow:
            closes = [bar.close for bar in bars]
            previous_fast = sum(closes[-self.fast - 1 : -1]) / self.fast
            previous_slow = sum(closes[-self.slow - 1 : -1]) / self.slow
            current_fast = sum(closes[-self.fast :]) / self.fast
            current_slow = sum(closes[-self.slow :]) / self.slow
            if previous_fast <= previous_slow and current_fast > current_slow:
                side = "buy"
            elif previous_fast >= previous_slow and current_fast < current_slow:
                side = "sell"
            explanation = f"Demo SMA({self.fast},{self.slow}) crossover: {side}; no profit claim"
        return TradeSignal(
            symbol=bars[-1].instrument.symbol,
            side=side,
            timestamp=bars[-1].timestamp,
            explanation=explanation,
        )
