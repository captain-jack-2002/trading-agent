"""Explicit typed adapters. Free-form retrieved text never creates authoritative facts."""

import json
from datetime import datetime
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field

from trading_agent.config.settings import Settings
from trading_agent.integrations.nse_mcp.client import MCPResearchContext
from trading_agent.models.domain import (
    DomainModel,
    ExecutableMarketQuote,
    OrderResult,
    PortfolioSnapshot,
    Symbol,
)

from .contracts import FactClass, FactObservation, SourceType, ToolEvidence, content_hash

Finite = Annotated[float, Field(allow_inf_nan=False)]


class ModelInference(DomainModel):
    """Adapter DTO supplied by model inference code, never parsed from model prose."""

    model_id: Annotated[str, Field(min_length=1)]
    symbol: Symbol
    timestamp: AwareDatetime
    probability: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] | None = None
    score: Finite | None = None


class ModelMetric(DomainModel):
    """A measured value read from trusted registry/report metadata by the caller."""

    model_id: Annotated[str, Field(min_length=1)]
    metric: Annotated[str, Field(min_length=1)]
    value: Finite
    source_id: Annotated[str, Field(min_length=1)]
    timestamp: AwareDatetime
    source_type: Literal[SourceType.REGISTRY_METADATA, SourceType.PHASE_REPORT] = (
        SourceType.REGISTRY_METADATA
    )


class MarketFields(DomainModel):
    """Licensed provider fields; does not accept NSE MCP response objects."""

    symbol: Symbol
    provider: Annotated[str, Field(min_length=1)]
    timestamp: AwareDatetime
    volume: Annotated[int, Field(ge=0, strict=True)] | None = None
    open_interest: Annotated[int, Field(ge=0, strict=True)] | None = None
    implied_volatility: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None


def _number(value: Decimal | float | int) -> str:
    return format(Decimal(str(value)).normalize(), "f")


def _fact(fact_class: FactClass, key: str, value: Decimal | float | int | str) -> FactObservation:
    return FactObservation(
        fact_class=fact_class, key=key, value=value if isinstance(value, str) else _number(value)
    )


def _typed(
    source_id: str,
    source_type: SourceType,
    timestamp: datetime,
    content: str,
    facts: tuple[FactObservation, ...],
) -> ToolEvidence:
    evidence = ToolEvidence(
        source_id=source_id, source_type=source_type, source_timestamp=timestamp, content=content
    )
    object.__setattr__(evidence, "_facts", facts)
    object.__setattr__(evidence, "_binding", content_hash(evidence.model_dump_json()))
    return evidence


def evidence_from_quote(value: ExecutableMarketQuote) -> ToolEvidence:
    if not isinstance(value, ExecutableMarketQuote):
        raise TypeError("executable quote adapter requires ExecutableMarketQuote")
    value = ExecutableMarketQuote.model_validate(value.model_dump())
    quote = value.quote
    return _typed(
        quote.source,
        SourceType.EXECUTABLE_QUOTE,
        quote.timestamp,
        value.model_dump_json(),
        (_fact(FactClass.EXECUTABLE_PRICE, quote.instrument.symbol, quote.price),),
    )


def evidence_from_portfolio(
    value: PortfolioSnapshot, *, source_id: str = "paper_ledger"
) -> ToolEvidence:
    if not isinstance(value, PortfolioSnapshot):
        raise TypeError("ledger adapter requires PortfolioSnapshot")
    value = PortfolioSnapshot.model_validate(value.model_dump())
    facts = (
        _fact(FactClass.ACCOUNT_BALANCE, "cash", value.cash),
        _fact(FactClass.ACCOUNT_EQUITY, "equity", value.equity),
    ) + tuple(
        _fact(FactClass.OPEN_POSITION, position.symbol, position.quantity)
        for position in value.positions
    )
    return _typed(source_id, SourceType.LEDGER, value.as_of, value.model_dump_json(), facts)


def evidence_from_order(
    value: OrderResult, *, as_of: datetime, source_id: str = "paper_order_adapter"
) -> ToolEvidence:
    if not isinstance(value, OrderResult):
        raise TypeError("order adapter requires OrderResult")
    value = OrderResult.model_validate(value.model_dump())
    return _typed(
        source_id,
        SourceType.ORDER_ADAPTER,
        as_of,
        value.model_dump_json(),
        (_fact(FactClass.ORDER_STATE, value.client_order_id, value.status),),
    )


def evidence_from_inference(value: ModelInference) -> ToolEvidence:
    if not isinstance(value, ModelInference):
        raise TypeError("inference adapter requires ModelInference")
    value = ModelInference.model_validate(value.model_dump())
    key = f"{value.model_id}:{value.symbol}"
    facts = []
    if value.probability is not None:
        facts.append(_fact(FactClass.MODEL_PROBABILITY, key, value.probability))
    if value.score is not None:
        facts.append(_fact(FactClass.MODEL_SCORE, key, value.score))
    return _typed(
        f"model:{value.model_id}",
        SourceType.MODEL_INFERENCE,
        value.timestamp,
        value.model_dump_json(),
        tuple(facts),
    )


def evidence_from_metric(value: ModelMetric) -> ToolEvidence:
    if not isinstance(value, ModelMetric):
        raise TypeError("metric adapter requires ModelMetric")
    value = ModelMetric.model_validate(value.model_dump())
    return _typed(
        value.source_id,
        value.source_type,
        value.timestamp,
        value.model_dump_json(),
        (_fact(FactClass.MODEL_METRIC, f"{value.model_id}:{value.metric}", value.value),),
    )


def evidence_from_market_fields(value: MarketFields) -> ToolEvidence:
    if not isinstance(value, MarketFields):
        raise TypeError("market fields adapter requires MarketFields")
    value = MarketFields.model_validate(value.model_dump())
    facts = []
    if value.volume is not None:
        facts.append(_fact(FactClass.MARKET_VOLUME, value.symbol, value.volume))
    if value.open_interest is not None:
        facts.append(_fact(FactClass.OPEN_INTEREST, value.symbol, value.open_interest))
    if value.implied_volatility is not None:
        facts.append(
            _fact(
                FactClass.DERIVATIVE_FIELD,
                f"{value.symbol}:implied_volatility",
                value.implied_volatility,
            )
        )
    return _typed(
        value.provider,
        SourceType.MARKET_DATA,
        value.timestamp,
        value.model_dump_json(),
        tuple(facts),
    )


def evidence_from_risk_settings(value: Settings, *, as_of: datetime) -> ToolEvidence:
    if not isinstance(value, Settings):
        raise TypeError("risk settings adapter requires Settings")
    # Whitelist rather than serializing Settings, which can contain connection secrets.
    limits = {
        "max_capital_per_trade": _number(value.max_capital_per_trade),
        "max_portfolio_exposure": _number(value.max_portfolio_exposure),
        "max_open_positions": str(value.max_open_positions),
        "max_daily_loss": _number(value.max_daily_loss),
        "quote_max_age_seconds": str(value.quote_max_age_seconds),
        "mandatory_stop_loss": str(value.mandatory_stop_loss).lower(),
    }
    return _typed(
        "risk_settings",
        SourceType.RISK_CONFIG,
        as_of,
        json.dumps(limits, sort_keys=True),
        tuple(_fact(FactClass.RISK_LIMIT, key, val) for key, val in limits.items()),
    )


def evidence_from_mcp(value: MCPResearchContext) -> ToolEvidence | None:
    if not isinstance(value, MCPResearchContext):
        raise TypeError("MCP adapter requires MCPResearchContext")
    if value.status != "available" or value.source is None or value.data is None:
        return None
    return ToolEvidence(
        source_id=f"nse_mcp:{value.source.source_server}:{value.source.tool_name}",
        source_type=SourceType.NSE_MCP,
        source_timestamp=value.source.retrieval_timestamp,
        content=json.dumps(value.data, sort_keys=True, allow_nan=False),
    )
