from datetime import time
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PositiveMoney = Annotated[Decimal, Field(gt=0, allow_inf_nan=False)]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRADING_", frozen=True)

    execution_mode: Literal["paper"] = "paper"
    database_url: str = "postgresql+psycopg://paper@localhost:5432/paper"
    valkey_url: str | None = None
    initial_cash: PositiveMoney = Decimal("100000")
    max_capital_per_trade: PositiveMoney = Decimal("10000")
    max_portfolio_exposure: PositiveMoney = Decimal("50000")
    max_open_positions: Annotated[int, Field(gt=0)] = 5
    max_daily_loss: PositiveMoney = Decimal("5000")
    allowed_symbols: frozenset[str] = frozenset({"DEMO"})
    allowed_instrument_types: frozenset[str] = frozenset({"equity"})
    quote_max_age_seconds: Annotated[int, Field(gt=0)] = 30
    mandatory_stop_loss: bool = True
    market_open: time = time(9, 15)
    market_close: time = time(15, 30)
    holidays: frozenset[str] = frozenset()
    nse_mcp_enabled: bool = True
    nse_bhavcopy_mcp_url: str = Field(
        default="https://mcp.nseindia.in/bhavcopy/cm/mcp",
        validation_alias=AliasChoices("NSE_BHAVCOPY_MCP_URL", "TRADING_NSE_BHAVCOPY_MCP_URL"),
    )
    nse_cm_market_mcp_url: str = Field(
        default="https://mcp.nseindia.in/cmmkt/mcp",
        validation_alias=AliasChoices("NSE_CM_MARKET_MCP_URL", "TRADING_NSE_CM_MARKET_MCP_URL"),
    )
    nse_mcp_timeout_seconds: Annotated[float, Field(gt=0, le=120)] = 10.0
    nse_mcp_cache_ttl_seconds: Annotated[int, Field(gt=0, le=3600)] = 60

    @model_validator(mode="after")
    def validate_session(self) -> "Settings":
        if self.market_open >= self.market_close:
            raise ValueError("market_open must precede market_close")
        return self
