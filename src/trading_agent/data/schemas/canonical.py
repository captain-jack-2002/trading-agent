"""Provider-neutral immutable data records. No exchange metadata is inferred."""

from datetime import UTC, date, datetime
from typing import Annotated, Literal
from zoneinfo import ZoneInfo

from pydantic import AwareDatetime, Field, field_validator, model_validator

from trading_agent.models.domain import DomainModel, PositiveMoney

NonnegativeInt = Annotated[int, Field(ge=0)]
Name = Annotated[str, Field(min_length=1, pattern=r"^\S(?:.*\S)?$")]


class Contract(DomainModel):
    contract_id: Name
    instrument_type: Literal["FUTSTK", "FUTIDX", "OPTSTK", "OPTIDX"]
    symbol: Name
    exchange: Name
    underlying: Name
    expiry: date
    lot_size: Annotated[int, Field(gt=0, strict=True)]
    tick_size: PositiveMoney
    strike: PositiveMoney | None = None
    option_type: Literal["CE", "PE"] | None = None

    @model_validator(mode="after")
    def validate_kind(self) -> "Contract":
        option = self.instrument_type.startswith("OPT")
        if option != (self.strike is not None and self.option_type is not None):
            raise ValueError("options require strike and option_type")
        if not option and (self.strike is not None or self.option_type is not None):
            raise ValueError("futures cannot carry option fields")
        return self


class CanonicalBar(DomainModel):
    asset_class: Literal["equity", "future", "option"]
    symbol: Name
    exchange: Name
    timestamp: AwareDatetime
    open: PositiveMoney
    high: PositiveMoney
    low: PositiveMoney
    close: PositiveMoney
    volume: NonnegativeInt
    underlying: Name | None = None
    expiry: date | None = None
    strike: PositiveMoney | None = None
    option_type: Literal["CE", "PE"] | None = None
    open_interest: NonnegativeInt | None = None
    contract: Contract | None = None

    @field_validator("timestamp")
    @classmethod
    def utc_timestamp(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_bar(self) -> "CanonicalBar":
        if not self.low <= min(self.open, self.close) <= max(self.open, self.close) <= self.high:
            raise ValueError("inconsistent OHLC range")
        if self.asset_class == "equity":
            if any(
                x is not None
                for x in (
                    self.underlying,
                    self.expiry,
                    self.strike,
                    self.option_type,
                    self.open_interest,
                    self.contract,
                )
            ):
                raise ValueError("equity cannot carry derivative metadata")
        else:
            if self.underlying is None or self.expiry is None:
                raise ValueError("derivatives require underlying and expiry")
            if self.timestamp.astimezone(ZoneInfo("Asia/Kolkata")).date() > self.expiry:
                raise ValueError("bar occurs after contract expiry")
            if self.asset_class == "option":
                if self.strike is None or self.option_type is None:
                    raise ValueError("options require strike and option_type")
            elif self.strike is not None or self.option_type is not None:
                raise ValueError("futures cannot carry option fields")
            if self.contract is not None:
                c = self.contract
                expected = "option" if c.instrument_type.startswith("OPT") else "future"
                if (
                    self.asset_class,
                    self.symbol,
                    self.exchange,
                    self.underlying,
                    self.expiry,
                    self.strike,
                    self.option_type,
                ) != (
                    expected,
                    c.symbol,
                    c.exchange,
                    c.underlying,
                    c.expiry,
                    c.strike,
                    c.option_type,
                ):
                    raise ValueError("contract reference disagrees with bar identity")
                if any(
                    price % c.tick_size != 0
                    for price in (self.open, self.high, self.low, self.close)
                ):
                    raise ValueError("price does not align with supplied contract tick_size")
        return self

    @property
    def instrument_key(self) -> tuple[str, ...]:
        return tuple(
            str(x) if x is not None else ""
            for x in (
                self.asset_class,
                self.exchange,
                self.symbol,
                self.underlying,
                self.expiry,
                self.strike.normalize() if self.strike is not None else None,
                self.option_type,
            )
        )
