"""Local model proposals with fresh drift, lifecycle and independent paper risk gates."""

import math
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

from trading_agent.execution.paper import PaperBrokerAdapter
from trading_agent.features.research import FeatureRow
from trading_agent.grounding import GroundingRequest, GroundingService
from trading_agent.ml.audit import require_synthetic
from trading_agent.ml.drift import DriftThresholds, evaluate_drift
from trading_agent.ml.lifecycle import LifecycleRegistry, digest
from trading_agent.ml.pipeline import predict_probabilities, predict_returns
from trading_agent.ml.registry import load_model, show_model
from trading_agent.models.decision import DecisionRecord
from trading_agent.models.domain import OrderRequest
from trading_agent.persistence.tables import AuditRecord


class ModelPaperExecutor:
    """Caller supplies feature observations and sizing intent, never a model score.

    A challenger may produce shadow/paper orders; only explicit promotion changes
    the champion. No live adapter, cloud service, or retraining is invoked here.
    """

    def __init__(self, broker: PaperBrokerAdapter, model: Path, *, trusted: bool = False):
        if not trusted:
            raise ValueError("explicit trust required for local executable model artifact")
        self.broker = broker
        self.model_path = model.resolve()
        self.registry = LifecycleRegistry(self.model_path.parent)

    def _abstain(self, record: DecisionRecord, reason: str) -> DecisionRecord:
        record = record.model_copy(update={"outcome": "abstained", "reasons": (reason,)})
        with self.broker.store.transaction() as session:
            session.add(
                AuditRecord(
                    created_at=record.event_timestamp,
                    payload={
                        "event": "model_decision",
                        "decision": record.model_dump(mode="json"),
                    },
                )
            )
        return record

    def submit(
        self,
        rows: Sequence[FeatureRow],
        order: OrderRequest,
        *,
        probability_threshold: float = 0.5,
        return_threshold: float = 0.0,
        thresholds: DriftThresholds | None = None,
        grounding_request: GroundingRequest | None = None,
        grounding_service: GroundingService | None = None,
    ) -> DecisionRecord:
        if not math.isfinite(probability_threshold) or not 0 < probability_threshold < 1:
            raise ValueError("probability threshold must be in (0,1)")
        if not math.isfinite(return_threshold):
            raise ValueError("return threshold must be finite")
        order = OrderRequest.model_validate(order.model_dump())
        now = self.broker.clock()
        # Validate/snapshot mutable FeatureRow.values before hashing and inference.
        record = DecisionRecord(
            decision_id=str(uuid4()),
            instrument=order.symbol,
            event_timestamp=now,
            dataset_versions=tuple(sorted({r.dataset_version for r in rows})),
            feature_version=rows[-1].feature_version if rows else "unknown",
            features_sha256=digest([]),
            model_id=self.model_path.name,
            model_version="unknown",
            model_artifact_sha256="unknown",
            prediction_kind="positive_probability",
            model_health="quarantined",
            proposal=order,
        )
        try:
            rows = [FeatureRow.model_validate(r.model_dump()) for r in rows]
            for row in rows:
                require_synthetic(row.provenance, row.synthetic)
            record = record.model_copy(
                update={
                    "features_sha256": digest([r.model_dump(mode="json") for r in rows]),
                    "feature_timestamp": rows[-1].timestamp if rows else None,
                }
            )
            if grounding_request is not None:
                request = grounding_request.model_copy(update={"as_of": now})
                grounded = (grounding_service or GroundingService()).query(request)
                record = record.model_copy(
                    update={
                        "evidence_ids": tuple(e.evidence_id for e in grounded.evidence),
                        "source_hashes": tuple(e.provenance.source_hash for e in grounded.evidence),
                        "fact_decisions": grounded.decisions,
                        "fact_policy_sha256": digest(
                            [d.model_dump(mode="json") for d in grounded.decisions]
                        ),
                        "unsupported_claims": tuple(
                            d.requirement.key for d in grounded.decisions if d.status != "supported"
                        )
                        or grounded.reasons
                        if grounded.status == "abstained"
                        else (),
                    }
                )
                if grounded.status == "abstained":
                    return self._abstain(record, "insufficient_grounding_evidence")
            self.registry.require_usable(self.model_path.name)
            bundle = load_model(self.model_path, trusted=True)
            record = record.model_copy(
                update={
                    "model_id": str(bundle.metadata["model_id"]),
                    "model_version": str(bundle.metadata["model_version"]),
                    "model_metadata_sha256": self.registry._artifact(self.model_path.name)[1],
                    "model_artifact_sha256": show_model(
                        self.model_path.parent, self.model_path.name
                    )["metadata"]["sha256"],
                    "prediction_kind": "future_return"
                    if bundle.metadata.get("task") == "regression"
                    else "positive_probability",
                }
            )
            # A complete ordered, instrument-local monitoring window is required.
            if (
                not rows
                or any(r.instrument_id != order.symbol for r in rows)
                or any(a.timestamp >= b.timestamp for a, b in zip(rows, rows[1:], strict=False))
                or rows[-1].timestamp > now
                or (now - rows[-1].timestamp).total_seconds()
                > self.broker.settings.quote_max_age_seconds
            ):
                return self._abstain(record, "invalid_or_stale_feature_window")
            cutoff = bundle.metadata["ranges"]["validation"]["label_end"]
            from datetime import datetime

            if rows[0].timestamp <= datetime.fromisoformat(cutoff):
                return self._abstain(record, "model_deployment_chronology_violation")
            self.registry.require_usable(self.model_path.name)
            health = evaluate_drift(bundle.metadata, rows, thresholds=thresholds)
            if not health.allows_signals:
                self.registry.quarantine(
                    self.model_path.name,
                    reason="drift:" + digest(health.model_dump(mode="json")),
                    actor="drift-monitor",
                )
                record = record.model_copy(
                    update={
                        "model_health": health.status,
                        "health": health,
                        "drift_reference": digest(health.model_dump(mode="json")),
                    }
                )
                return self._abstain(record, "model_health_quarantined")
            predictions = (
                predict_returns(bundle, rows)
                if bundle.metadata.get("task") == "regression"
                else predict_probabilities(bundle, rows)
            )
            health = evaluate_drift(
                bundle.metadata, rows, predictions=predictions, thresholds=thresholds
            )
            record = record.model_copy(
                update={
                    "prediction": predictions[-1],
                    "model_health": health.status,
                    "health": health,
                    "drift_reference": digest(health.model_dump(mode="json")),
                }
            )
            if not health.allows_signals:
                self.registry.quarantine(
                    self.model_path.name,
                    reason="drift:" + str(record.drift_reference),
                    actor="drift-monitor",
                )
                return self._abstain(record, "model_health_quarantined")
            positive = (
                predictions[-1] > return_threshold
                if bundle.metadata.get("task") == "regression"
                else predictions[-1] >= probability_threshold
            )
            if (order.side == "buy") != positive:
                return self._abstain(record, "model_does_not_support_proposed_side")
            with self.registry.signal_gate(self.model_path.name) as state:
                record = record.model_copy(update={"lifecycle_reference": state.event_hash})
                _, final = self.broker._submit(order, record)
                assert final is not None
                return final
        except (ValueError, KeyError, OSError, TypeError):
            # Do not echo arbitrary model/provider exception text into audit records.
            return self._abstain(record, "model_or_registry_invalid_or_quarantined")
