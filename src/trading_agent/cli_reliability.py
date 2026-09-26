"""Explicit offline reliability commands; inference requires trusted local artifacts."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from trading_agent.features.research import FeatureRow
from trading_agent.grounding import (
    GroundingRequest,
    LocalGroundingIndex,
    SourceType,
    load_documents,
    query_index,
)
from trading_agent.ml.dataset import DatasetRow
from trading_agent.ml.drift import (
    DriftDiagnostic,
    DriftThresholds,
    ModelHealth,
    MonitoringWindow,
    RealizedLabel,
    evaluate_drift,
)
from trading_agent.ml.lifecycle import LifecycleRegistry, PromotionEvidence, digest
from trading_agent.ml.registry import load_model, show_model
from trading_agent.research_io import write_json


def add_commands(groups: Any, model_actions: Any) -> None:
    grounding = groups.add_parser("grounding", help="Local evidence retrieval and provenance")
    actions = grounding.add_subparsers(dest="action", required=True)
    index = actions.add_parser("index")
    index.add_argument("--source", type=Path, required=True)
    index.add_argument("--output", type=Path, required=True)
    index.add_argument("--registry", type=Path)
    index.add_argument("--trusted-artifact", type=Path, action="append", default=[])
    index.add_argument("--chunk-words", type=int, default=180)
    query = actions.add_parser("query")
    query.add_argument("query")
    query.add_argument("--index", type=Path, required=True)
    query.add_argument("--as-of", type=datetime.fromisoformat)
    query.add_argument("--top-k", type=int, default=5)
    query.add_argument("--min-score", type=float, default=0.1)
    query.add_argument("--source-type", type=SourceType, action="append", default=[])
    query.add_argument("--source-id", action="append", default=[])
    query.add_argument("--max-age-seconds", type=float)
    inspect = actions.add_parser("inspect")
    inspect.add_argument("evidence_id")
    inspect.add_argument("--index", type=Path, required=True)
    for name in (
        "health",
        "drift",
        "promote",
        "quarantine",
        "retire",
        "enroll",
        "challenge",
        "audit",
    ):
        command = model_actions.add_parser(name)
        command.add_argument("--registry", type=Path, default=Path("data/models"))
        if name != "audit":
            command.add_argument("model_id")
        if name in ("health", "drift"):
            command.add_argument("--window", type=Path, required=name == "drift")
            command.add_argument("--thresholds", type=Path)
            command.add_argument("--trust-local-artifact", action="store_true")
        if name in ("quarantine", "retire", "enroll", "challenge"):
            command.add_argument("--reason", required=True)
            command.add_argument("--actor", default="local-operator")
        if name == "promote":
            command.add_argument("--walk-forward", type=Path, required=True)
            command.add_argument("--paper-validation", type=Path, required=True)
            command.add_argument("--approved-by", required=True)
            command.add_argument("--reason", required=True)


def grounding_command(args: argparse.Namespace) -> dict[str, Any]:
    if args.action == "index":
        documents = load_documents(
            args.source, registry=args.registry, trusted_artifacts=tuple(args.trusted_artifact)
        )
        if not args.source.exists():
            raise ValueError("grounding source does not exist")
        index = LocalGroundingIndex.from_documents(documents, chunk_words=args.chunk_words)
        index.save(args.output)
        return {
            "documents": len(index.documents),
            "chunks": len(index.chunks),
            "output": str(args.output),
        }
    if args.action == "inspect":
        chunk = LocalGroundingIndex.load(args.index).inspect(args.evidence_id)
        if chunk is None:
            raise ValueError("evidence ID not found in local index")
        return {"chunk": chunk.model_dump(mode="json")}
    return query_index(
        args.index,
        GroundingRequest(
            query=args.query,
            as_of=args.as_of or datetime.now(UTC),
            top_k=args.top_k,
            min_score=args.min_score,
            source_types=tuple(args.source_type),
            source_ids=tuple(args.source_id),
            max_age_seconds=args.max_age_seconds,
        ),
    ).model_dump(mode="json")


def _window(payload: Any) -> MonitoringWindow:
    if not isinstance(payload, dict) or set(payload) - {
        "schema_version",
        "rows",
        "labels",
        "evaluated_at",
    }:
        raise ValueError("invalid monitoring artifact fields; predictions must be inferred")
    if payload.get("schema_version", "monitoring-window-v1") != "monitoring-window-v1":
        raise ValueError("unsupported monitoring window version")
    rows = tuple(
        (DatasetRow if "label_end" in row else FeatureRow).model_validate(row)
        for row in payload["rows"]
    )
    return MonitoringWindow(
        rows=rows,
        evaluated_at=payload.get("evaluated_at", datetime.now(UTC)),
        # Label validation is completed after independently computing aligned predictions.
    )


def model_command(args: argparse.Namespace) -> dict[str, Any]:
    registry = LifecycleRegistry(args.registry)
    if args.action == "audit":
        return {"events": [e.model_dump(mode="json") for e in registry.events()]}
    if args.action == "promote":
        evidence = PromotionEvidence(
            walk_forward_path=args.walk_forward.resolve(),
            paper_validation_path=args.paper_validation.resolve(),
            walk_forward_sha256=hashlib.sha256(args.walk_forward.read_bytes()).hexdigest(),
            paper_validation_sha256=hashlib.sha256(args.paper_validation.read_bytes()).hexdigest(),
            approved_by=args.approved_by,
            reason=args.reason,
        )
        return registry.promote(args.model_id, evidence).model_dump(mode="json")
    if args.action in ("quarantine", "retire", "enroll", "challenge"):
        operation = getattr(registry, args.action)
        return dict(
            operation(args.model_id, actor=args.actor, reason=args.reason).model_dump(mode="json")
        )
    metadata = show_model(args.registry, args.model_id)["metadata"]
    state = registry.state(args.model_id)
    if args.window is None:
        return {
            "lifecycle": state.model_dump(mode="json"),
            "health": None,
            "reason": "Health not evaluated: an observation window is required",
        }
    policy = (
        DriftThresholds.model_validate_json(args.thresholds.read_text())
        if args.thresholds
        else DriftThresholds()
    )
    raw_window = args.window.read_bytes()
    try:
        payload = json.loads(raw_window)
        window = _window(payload)
        health = evaluate_drift(metadata, window, thresholds=policy)
    except (ValueError, TypeError, KeyError, ValidationError):
        window = None
        health = ModelHealth(
            status="quarantined",
            window_count=0,
            model_id=str(metadata["model_id"]),
            threshold_version=policy.version,
            diagnostics=(
                DriftDiagnostic(
                    category="data_quality",
                    metric="window_integrity",
                    severity="quarantined",
                    reason="Invalid monitoring artifact",
                ),
            ),
        )
    if health.allows_signals and window is not None:
        if not args.trust_local_artifact:
            raise ValueError("prediction monitoring requires --trust-local-artifact")
        from trading_agent.ml.pipeline import predict_probabilities, predict_returns

        bundle = load_model(args.registry / args.model_id, trusted=True)
        predictions = (
            predict_returns(bundle, window.rows)
            if metadata["task"] == "regression"
            else predict_probabilities(bundle, window.rows)
        )
        try:
            labels = (
                tuple(RealizedLabel.model_validate(label) for label in payload["labels"])
                if payload.get("labels") is not None
                else None
            )
            monitored = MonitoringWindow(
                rows=window.rows,
                evaluated_at=window.evaluated_at,
                predictions=tuple(predictions),
                labels=labels,
            )
            health = evaluate_drift(metadata, monitored, thresholds=policy)
        except (ValueError, TypeError, KeyError, ValidationError):
            health = ModelHealth(
                status="quarantined",
                window_count=len(window.rows),
                model_id=str(metadata["model_id"]),
                threshold_version=policy.version,
                diagnostics=(
                    DriftDiagnostic(
                        category="data_quality",
                        metric="realized_labels",
                        severity="quarantined",
                        reason="Invalid realized label evidence",
                    ),
                ),
            )
    reference = digest(health.model_dump(mode="json"))
    # Every evaluation is an immutable report. Quarantine latches; healthy never clears it.
    report = {
        "model_id": metadata["model_id"],
        "health": health.model_dump(mode="json"),
        "window_sha256": hashlib.sha256(raw_window).hexdigest(),
        "evaluated_at": window.evaluated_at.isoformat() if window else None,
        "thresholds": policy.model_dump(mode="json"),
    }
    report_path = args.registry / "monitoring" / (digest(report) + ".json")
    try:
        write_json(report_path, report)
    except FileExistsError:
        if report_path.is_symlink() or json.loads(report_path.read_text()) != report:
            raise ValueError("monitoring report integrity mismatch") from None
    if not health.allows_signals and state.status != "quarantined":
        registry.quarantine(args.model_id, reason="drift:" + reference, actor="drift-monitor")
    return {
        "health": health.model_dump(mode="json"),
        "lifecycle": registry.state(args.model_id).model_dump(mode="json"),
        "health_reference": reference,
        "report": str(report_path),
    }
