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
