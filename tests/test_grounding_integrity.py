"""Evidence tampering and persistence failures must never confer authority."""

from datetime import UTC, datetime
from decimal import Decimal

from trading_agent.grounding import (
    FactClass,
    FactRequirement,
    GroundingRequest,
    GroundingService,
    evidence_from_quote,
)
from trading_agent.models.domain import ExecutableMarketQuote, Instrument, Quote

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def test_copying_adapter_output_cannot_change_authoritative_content():
    evidence = evidence_from_quote(
        ExecutableMarketQuote(
            quote=Quote(
                instrument=Instrument(symbol="TEST"),
                price=Decimal(100),
                timestamp=NOW,
            )
        )
    )
    forged = evidence.model_copy(update={"content": "TEST price 1000000"})
    result = GroundingService().query(
        GroundingRequest(
            query="price",
            as_of=NOW,
            tool_evidence=(forged,),
            fact_requirements=(FactRequirement(fact_class=FactClass.EXECUTABLE_PRICE, key="TEST"),),
        )
    )
    assert result.status == "abstained"
    assert not result.evidence[0].provenance.executable_price


def test_quote_adapter_revalidates_bypassed_negative_price():
    import pytest

    value = ExecutableMarketQuote(
        quote=Quote(
            instrument=Instrument(symbol="TEST"),
            price=Decimal(100),
            timestamp=NOW,
        )
    )
    forged = value.model_copy(
        update={"quote": value.quote.model_copy(update={"price": Decimal(-1)})}
    )
    with pytest.raises(ValueError):
        evidence_from_quote(forged)


def test_known_mcp_source_cannot_be_cast_into_executable_evidence():
    import pytest

    quote = ExecutableMarketQuote(
        quote=Quote(
            instrument=Instrument(symbol="TEST"),
            price=Decimal(100),
            timestamp=NOW,
            source="NSE_MCP",
        )
    )
    with pytest.raises(ValueError, match="informational"):
        evidence_from_quote(quote)


def test_known_mcp_provider_cannot_supply_trading_market_fields():
    import pytest

    from trading_agent.grounding import MarketFields

    with pytest.raises(ValueError, match="informational"):
        MarketFields(symbol="TEST", provider="NSE_MCP", timestamp=NOW, open_interest=100)
