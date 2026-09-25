"""Explicit research cost assumptions; no embedded official fee schedule."""

from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

NonNegative = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]
Side = Literal["buy", "sell"]
Product = Literal["equity_delivery", "equity_intraday", "future", "option"]
ZERO = Decimal("0")


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Rate(Frozen):
    buy: NonNegative = ZERO
    sell: NonNegative = ZERO
    cap: NonNegative | None = None

    def amount(self, side: Side, turnover: Decimal) -> Decimal:
        value = turnover * (self.buy if side == "buy" else self.sell)
        return min(value, self.cap) if self.cap is not None else value


class CostBreakdown(Frozen):
    brokerage: NonNegative = ZERO
    stt: NonNegative = ZERO
    exchange: NonNegative = ZERO
    sebi: NonNegative = ZERO
    stamp: NonNegative = ZERO
    gst: NonNegative = ZERO

    @property
    def total(self) -> Decimal:
        return self.brokerage + self.stt + self.exchange + self.sebi + self.stamp + self.gst


class CostSchedule(Frozen):
    brokerage: Rate = Rate()
    stt: Rate = Rate()
    exchange: Rate = Rate()
    sebi: Rate = Rate()
    stamp: Rate = Rate()
    gst_rate: NonNegative = ZERO
    gst_components: tuple[Literal["brokerage", "exchange", "sebi"], ...] = (
        "brokerage",
        "exchange",
        "sebi",
    )

    def calculate(self, side: Side, price: Decimal, quantity: int) -> CostBreakdown:
        if not price.is_finite() or price <= 0 or quantity <= 0:
            raise ValueError("positive finite price and quantity required")
        turnover = price * quantity
        amounts = {
            name: getattr(self, name).amount(side, turnover)
            for name in ("brokerage", "stt", "exchange", "sebi", "stamp")
        }
        amounts["gst"] = sum((amounts[n] for n in set(self.gst_components)), ZERO) * self.gst_rate
        return CostBreakdown(**amounts)


class Slippage(Frozen):
    bps: NonNegative = ZERO
    percentage: NonNegative = ZERO
    volatility_multiplier: NonNegative = ZERO

    def price(self, price: Decimal, side: Side, volatility: Decimal = ZERO) -> Decimal:
        if not volatility.is_finite() or volatility < 0:
            raise ValueError("nonnegative finite historical volatility required")
        fraction = self.bps / Decimal(10000) + self.percentage
        fraction += self.volatility_multiplier * volatility
        result = price * (1 + fraction if side == "buy" else 1 - fraction)
        if not result.is_finite() or result <= 0:
            raise ValueError("slippage produces invalid price")
        return result
