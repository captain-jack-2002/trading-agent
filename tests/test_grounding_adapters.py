from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from trading_agent.config.settings import Settings
from trading_agent.grounding import (
    FactClass,
    FactRequirement,
    GroundingRequest,
    GroundingService,
    MarketFields,
    ModelInference,
    ModelMetric,
    SourceType,
    ToolEvidence,
    evidence_from_inference,
    evidence_from_market_fields,
    evidence_from_mcp,
    evidence_from_metric,
    evidence_from_order,
    evidence_from_portfolio,
    evidence_from_quote,
    evidence_from_risk_settings,
)
from trading_agent.integrations.nse_mcp.client import MCPResearchContext, ResearchSourceMetadata
from trading_agent.models.domain import (
    ExecutableMarketQuote,
    Instrument,
    OrderResult,
    PortfolioSnapshot,
    Position,
    Quote,
    RiskDecision,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def check(fact_class: FactClass, key: str, evidence: ToolEvidence):  # type: ignore[no-untyped-def]
    return GroundingService().query(
        GroundingRequest(
            query="facts",
            as_of=NOW,
            fact_requirements=(FactRequirement(fact_class=fact_class, key=key),),
            tool_evidence=(evidence,),
        )
    )


def test_freeform_source_label_and_serialized_fact_cannot_authorize_price() -> None:
    fake = ToolEvidence(
        source_id="liar",
        source_type=SourceType.EXECUTABLE_QUOTE,
        source_timestamp=NOW,
        content="TEST price 100",
    )
    result = check(FactClass.EXECUTABLE_PRICE, "TEST", fake)
    assert result.status == "abstained"
    assert not result.evidence[0].provenance.executable_price
    quote = ExecutableMarketQuote(
        quote=Quote(instrument=Instrument(symbol="TEST"), price=Decimal("100"), timestamp=NOW)
    )
    genuine = evidence_from_quote(quote)
    restored = ToolEvidence.model_validate_json(genuine.model_dump_json())
    assert check(FactClass.EXECUTABLE_PRICE, "TEST", restored).status == "abstained"
    with pytest.raises(TypeError):
        evidence_from_quote("TEST price 100")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        genuine._facts = ()


def test_portfolio_adapter_only_supports_observed_facts() -> None:
    portfolio = PortfolioSnapshot(
        cash=Decimal("1000"),
        equity=Decimal("1100"),
        day_start_equity=Decimal("1100"),
        as_of=NOW,
        positions=(
            Position(
                symbol="TEST", quantity=1, average_price=Decimal("90"), mark_price=Decimal("100")
            ),
        ),
    )
    evidence = evidence_from_portfolio(portfolio)
    assert check(FactClass.ACCOUNT_BALANCE, "cash", evidence).decisions[0].value == "1000"
    assert check(FactClass.ACCOUNT_EQUITY, "equity", evidence).decisions[0].value == "1100"
    assert check(FactClass.OPEN_POSITION, "TEST", evidence).decisions[0].value == "1"
    assert check(FactClass.OPEN_POSITION, "OTHER", evidence).status == "abstained"
    assert check(FactClass.EXECUTABLE_PRICE, "TEST", evidence).status == "abstained"


def test_mcp_is_ephemeral_informational_only() -> None:
    context = MCPResearchContext(
        status="available",
        data={"price": 100, "volume": 123},
        source=ResearchSourceMetadata(
            source_server="cm_market", tool_name="snapshot", retrieval_timestamp=NOW
        ),
    )
    evidence = evidence_from_mcp(context)
    assert evidence is not None
    result = check(FactClass.EXECUTABLE_PRICE, "TEST", evidence)
    assert result.status == "abstained"
    provenance = result.evidence[0].provenance
    assert provenance.source_id == "nse_mcp:cm_market:snapshot"
    assert provenance.informational_only
    assert not provenance.training_eligible
    assert not provenance.executable_price
    assert check(FactClass.MARKET_VOLUME, "TEST", evidence).status == "abstained"
    assert evidence_from_mcp(MCPResearchContext(status="research_context_unavailable")) is None
    research = GroundingService().query(
        GroundingRequest(query="snapshot", as_of=NOW, tool_evidence=(evidence,))
    )
    assert research.status == "supported"
    stale = GroundingService().query(
        GroundingRequest(
            query="snapshot", as_of=NOW + timedelta(seconds=301), tool_evidence=(evidence,)
        )
    )
    assert stale.status == "abstained"


def test_inference_adapter_scopes_probability_and_score_to_model_instrument() -> None:
    inference = ModelInference(
        model_id="model-1", symbol="TEST", probability=0.75, score=1.5, timestamp=NOW
    )
    evidence = evidence_from_inference(inference)
    assert check(FactClass.MODEL_PROBABILITY, "model-1:TEST", evidence).decisions[0].value == "0.75"
    assert check(FactClass.MODEL_SCORE, "model-1:TEST", evidence).decisions[0].value == "1.5"
    assert check(FactClass.MODEL_METRIC, "model-1:f1", evidence).status == "abstained"
    with pytest.raises(ValueError):
        ModelInference(model_id="bad", symbol="TEST", probability=1.1, timestamp=NOW)


def test_model_metrics_require_typed_metadata() -> None:
    evidence = evidence_from_metric(
        ModelMetric(
            model_id="model-1",
            metric="f1",
            value=0.6,
            source_id="data/models/model-1/metadata.json",
            timestamp=NOW,
        )
    )
    assert check(FactClass.MODEL_METRIC, "model-1:f1", evidence).decisions[0].value == "0.6"
    assert check(FactClass.MODEL_PROBABILITY, "model-1:TEST", evidence).status == "abstained"


def test_risk_adapter_snapshots_whitelisted_fields_without_secrets() -> None:
    evidence = evidence_from_risk_settings(Settings(database_url="secret-do-not-copy"), as_of=NOW)
    assert "secret-do-not-copy" not in evidence.content
    result = check(FactClass.RISK_LIMIT, "max_capital_per_trade", evidence)
    assert result.decisions[0].value == "10000"
    assert check(FactClass.RISK_LIMIT, "database_url", evidence).status == "abstained"


def test_order_adapter_and_market_fields_have_separate_authority() -> None:
    order = OrderResult(
        client_order_id="order1",
        status="rejected",
        risk=RiskDecision(
            approved=False, reasons=("risk rejected",), evaluated_at=NOW, client_order_id="order1"
        ),
    )
    evidence = evidence_from_order(order, as_of=NOW)
    assert check(FactClass.ORDER_STATE, "order1", evidence).decisions[0].value == "rejected"
    fields = evidence_from_market_fields(
        MarketFields(
            symbol="TEST",
            volume=123,
            open_interest=50,
            implied_volatility=0.2,
            timestamp=NOW,
            provider="licensed",
        )
    )
    assert check(FactClass.MARKET_VOLUME, "TEST", fields).decisions[0].value == "123"
    assert check(FactClass.OPEN_INTEREST, "TEST", fields).decisions[0].value == "50"
    assert (
        check(FactClass.DERIVATIVE_FIELD, "TEST:implied_volatility", fields).decisions[0].value
        == "0.2"
    )
    assert check(FactClass.EXECUTABLE_PRICE, "TEST", fields).status == "abstained"


def test_numerically_equal_prices_do_not_conflict() -> None:
    values = tuple(
        evidence_from_quote(
            ExecutableMarketQuote(
                quote=Quote(
                    instrument=Instrument(symbol="TEST"), price=Decimal(price), timestamp=NOW
                )
            )
        )
        for price in ("100", "100.00")
    )
    result = GroundingService().query(
        GroundingRequest(
            query="price",
            as_of=NOW,
            tool_evidence=values,
            fact_requirements=(FactRequirement(fact_class=FactClass.EXECUTABLE_PRICE, key="TEST"),),
        )
    )
    assert result.status == "supported"
