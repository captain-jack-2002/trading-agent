from datetime import date
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

Money = Annotated[Decimal, Field(allow_inf_nan=False)]
PositiveMoney = Annotated[Decimal, Field(gt=0, allow_inf_nan=False)]
Quantity = Annotated[int, Field(gt=0, strict=True)]
Symbol = Annotated[str, Field(pattern=r"^[A-Z0-9][A-Z0-9&._-]{0,29}$")]


class DomainModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Instrument(DomainModel):
    symbol: Symbol
    instrument_type: Literal["equity", "future", "option", "etf"] = "equity"
    exchange: Literal["NSE"] = "NSE"


class Quote(DomainModel):
    instrument: Instrument
    price: PositiveMoney
    timestamp: AwareDatetime
    source: str = "mock"


class OHLCVBar(DomainModel):
    instrument: Instrument
    timestamp: AwareDatetime
    open: PositiveMoney
    high: PositiveMoney
    low: PositiveMoney
    close: PositiveMoney
    volume: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def valid_range(self) -> "OHLCVBar":
        if not self.low <= min(self.open, self.close) <= max(self.open, self.close) <= self.high:
            raise ValueError("inconsistent OHLC range")
        return self


class TradeSignal(DomainModel):
    symbol: Symbol
    side: Literal["buy", "sell", "hold"]
    timestamp: AwareDatetime
    explanation: str
    strategy: str = "sma_crossover_demo"


class OrderRequest(DomainModel):
    client_order_id: Annotated[str, Field(min_length=1, max_length=100, pattern=r"^[\w.-]+$")]
    symbol: Symbol
    instrument_type: Literal["equity", "future", "option", "etf"] = "equity"
    side: Literal["buy", "sell"]
    quantity: Quantity
    stop_loss: PositiveMoney | None = None


class RiskDecision(DomainModel):
    approved: bool
    reasons: tuple[str, ...] = ()
    evaluated_at: AwareDatetime
    client_order_id: str


class OrderResult(DomainModel):
    client_order_id: str
    status: Literal["filled", "rejected"]
    filled_quantity: int = 0
    fill_price: PositiveMoney | None = None
    risk: RiskDecision


class Position(DomainModel):
    symbol: Symbol
    quantity: Quantity
    average_price: PositiveMoney
    mark_price: PositiveMoney

    @property
    def unrealized_pnl(self) -> Decimal:
        return (self.mark_price - self.average_price) * self.quantity


class PortfolioSnapshot(DomainModel):
    cash: Money
    equity: Money
    exposure: Money = Decimal("0")
    realized_pnl: Money = Decimal("0")
    unrealized_pnl: Money = Decimal("0")
    day_start_equity: Money
    trading_day: date | None = None
    positions: tuple[Position, ...] = ()
    as_of: AwareDatetime

    @property
    def daily_pnl(self) -> Decimal:
        return self.equity - self.day_start_equity
