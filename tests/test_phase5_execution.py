"""Synthetic engineering validation of high-confidence models and execution boundaries."""

from datetime import timedelta

import pytest
from pydantic import ValidationError
from sqlalchemy import select
from test_execution import broker as broker
from test_features_research import bars
from test_risk import NOW, order

from trading_agent.agent.reliability import ModelPaperExecutor
from trading_agent.features.research import FeatureRow
from trading_agent.ml.dataset import build_dataset
from trading_agent.ml.lifecycle import LifecycleRegistry
from trading_agent.ml.pipeline import predict_probabilities, train_model
from trading_agent.ml.registry import save_model
from trading_agent.ml.split import walk_forward
from trading_agent.models.domain import OrderRequest
from trading_agent.persistence.tables import FillRecord, OrderRecord


def test_direct_order_cannot_supply_model_or_risk_authority(broker):
    with pytest.raises(ValidationError):
        OrderRequest.model_validate({**order().model_dump(), "model_id": "challenger"})
    with pytest.raises(ValidationError):
        OrderRequest.model_validate({**order().model_dump(), "approved": True})
    assert broker.portfolio().cash == 100000


def test_manual_order_emits_hashed_decision_provenance(broker):
    result = broker.submit(order())
    assert result.status == "filled"
    records = [e for e in broker.store.audit() if e["event"] == "paper_order"]
    assert len(records) == 1
    record = records[0]["decision"]
    assert record["instrument"] == "DEMO"
    assert record["risk_decision"]["approved"] is True
    assert record["outcome"] == "filled"
    assert len(record["market_data_sha256"]) == 64
    assert record["decision_id"]


@pytest.mark.parametrize("untrusted", [{"price": "100", "source": "rag"}, "MCP price=100"])
def test_quote_cannot_be_replaced_by_free_form_or_mcp_output(broker, untrusted):
    class InformationalProvider:
        def quote(self, symbol):
            return untrusted

    broker.provider = InformationalProvider()
    result = broker.submit(order())
    assert result.status == "rejected"
    assert "quote_unavailable_or_invalid" in result.risk.reasons
    with broker.store.sessions() as session:
        assert not list(session.scalars(select(FillRecord)))


@pytest.fixture
def monitored(broker, tmp_path):
    rows = build_dataset(bars(150), horizon=2)
    split = walk_forward(rows, 70, 25, 25)[0]
    bundle = train_model(rows, split, model_kind="random_forest")
    model_path = tmp_path / "models" / "challenger"
    save_model(bundle, model_path)
    training = [rows[i] for i in split.train]
    scores = predict_probabilities(bundle, training)
    best = max(range(len(training)), key=lambda i: scores[i])
    assert scores[best] > 0.8
    ordered = [r for i, r in enumerate(training) if i != best] + [training[best]]
    window = tuple(
        FeatureRow.model_validate(
            {
                **{k: v for k, v in row.model_dump().items() if k in FeatureRow.model_fields},
                "instrument_id": "DEMO",
                "timestamp": NOW - timedelta(seconds=len(ordered) - i - 1),
            }
        )
        for i, row in enumerate(ordered)
    )
    executor = ModelPaperExecutor(broker, model_path, trusted=True)
    return broker, executor, window


def test_healthy_high_confidence_model_still_requires_deterministic_risk(monitored):
    broker, executor, window = monitored
    decision = executor.submit(window, order(quantity=101))
    assert decision.model_health == "healthy"
    assert decision.prediction > 0.8
    assert decision.outcome == "rejected"
    assert "capital_per_trade" in decision.risk_decision.reasons
    assert broker.portfolio().cash == 100000
    allowed = executor.submit(window, order(client_order_id="allowed"))
    assert allowed.outcome == "filled"
    assert allowed.risk_decision.approved
    assert allowed.evidence_ids and allowed.source_hashes and allowed.fact_policy_sha256


def test_quarantined_high_confidence_model_cannot_create_paper_order(monitored):
    broker, executor, window = monitored
    LifecycleRegistry(executor.model_path.parent).quarantine("challenger", reason="severe drift")
    result = executor.submit(window, order())
    assert result.outcome == "abstained"
    with broker.store.sessions() as session:
        assert not list(session.scalars(select(OrderRecord)))
        assert not list(session.scalars(select(FillRecord)))
    assert broker.portfolio().cash == 100000


def test_stale_feature_window_cannot_authorize(monitored):
    broker, executor, window = monitored
    stale = tuple(
        r.model_copy(update={"timestamp": r.timestamp - timedelta(hours=1)}) for r in window
    )
    assert executor.submit(stale, order()).outcome == "abstained"
    assert broker.portfolio().cash == 100000


def test_research_abstention_blocks_model_proposal(monitored):
    from trading_agent.grounding import FactClass, FactRequirement, GroundingRequest

    broker, executor, window = monitored
    request = GroundingRequest(
        query="model metric",
        as_of=NOW,
        fact_requirements=(
            FactRequirement(fact_class=FactClass.MODEL_METRIC, key="unmeasured:f1"),
        ),
    )
    result = executor.submit(window, order(), grounding_request=request)
    assert result.outcome == "abstained"
    assert result.unsupported_claims
    assert broker.portfolio().cash == 100000


def test_mcp_lineage_is_not_eligible_for_model_execution(monitored):
    broker, executor, window = monitored
    informational = tuple(
        row.model_copy(update={"provenance": "NSE_MCP informational only"}) for row in window
    )
    result = executor.submit(informational, order())
    assert result.outcome == "abstained"
    with broker.store.sessions() as session:
        assert not list(session.scalars(select(FillRecord)))


def test_typed_quote_with_known_mcp_lineage_is_rejected(broker):
    from decimal import Decimal

    from trading_agent.models.domain import Instrument, Quote

    class CastMCPProvider:
        def quote(self, symbol):
            return Quote(
                instrument=Instrument(symbol=symbol),
                price=Decimal(100),
                timestamp=NOW,
                source="NSE_MCP",
            )

    broker.provider = CastMCPProvider()
    result = broker.submit(order())
    assert result.status == "rejected"
    assert "quote_unavailable_or_invalid" in result.risk.reasons


def test_model_side_and_deployment_chronology_abstain(monitored):
    broker, executor, window = monitored
    side = executor.submit(window, order(side="sell"))
    assert side.outcome == "abstained"
    assert side.reasons == ("model_does_not_support_proposed_side",)
    from datetime import UTC, datetime

    historical = (
        window[0].model_copy(update={"timestamp": datetime(2000, 1, 1, tzinfo=UTC)}),
        *window[1:],
    )
    chronology = executor.submit(historical, order())
    assert chronology.outcome == "abstained"
    assert chronology.reasons == ("model_deployment_chronology_violation",)
    assert broker.portfolio().cash == 100000


def test_prediction_only_drift_latches_quarantine(monitored):
    broker, executor, window = monitored
    # Independently rotate feature columns: identical marginals, different joint relationships.
    columns = sorted(window[-1].values)
    scrambled = tuple(
        row.model_copy(
            update={
                "values": {
                    name: window[(i + j * 7) % len(window)].values[name]
                    for j, name in enumerate(columns)
                }
            }
        )
        for i, row in enumerate(window)
    )
    result = executor.submit(scrambled, order())
    assert result.health is not None
    assert all(
        d.severity == "healthy" for d in result.health.diagnostics if d.category == "feature"
    )
    assert any(
        d.severity == "quarantined" for d in result.health.diagnostics if d.category == "prediction"
    )
    assert result.outcome == "abstained"
    assert LifecycleRegistry(executor.model_path.parent).state("challenger").status == "quarantined"
    assert broker.portfolio().cash == 100000


def test_slow_inference_cannot_execute_with_expired_features_and_new_quote(monitored, monkeypatch):
    from trading_agent.agent import reliability
    from trading_agent.nse.mock import MockNSEProvider

    broker, executor, window = monitored
    current = [NOW]
    broker.clock = lambda: current[0]
    broker.provider = MockNSEProvider(clock=lambda: current[0])
    original = reliability.predict_probabilities

    def delayed_prediction(*args, **kwargs):
        prediction = original(*args, **kwargs)
        current[0] += timedelta(seconds=31)
        return prediction

    monkeypatch.setattr(reliability, "predict_probabilities", delayed_prediction)
    result = executor.submit(window, order())
    assert result.outcome == "rejected"
    assert "model_feature_window_stale" in result.risk_decision.reasons
    assert result.execution.fill_price is None
    assert broker.portfolio().cash == 100000
