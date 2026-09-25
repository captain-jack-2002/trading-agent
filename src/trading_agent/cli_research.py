"""Research CLI integration; only explicitly supplied local artifacts are read."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from trading_agent.data.schemas.canonical import CanonicalBar
from trading_agent.data.storage import ImportManifest, checksum, load_bars, verify_import
from trading_agent.features.research import ResearchBar, build_features
from trading_agent.ml.dataset import DatasetRow, build_dataset
from trading_agent.research_io import write_json


def read_import(path: Path) -> tuple[ImportManifest, list[CanonicalBar]]:
    if path.name != "manifest.json":
        raise ValueError("expected an imported dataset manifest.json")
    manifest = verify_import(path.parent)
    bars = list(load_bars(path.parent / "normalized.jsonl"))
    bars.sort(key=lambda bar: (bar.timestamp, bar.instrument_key))
    return manifest, bars


def research_bars(bars: list[CanonicalBar], manifest: ImportManifest) -> list[ResearchBar]:
    # Expiry DATE alone does not determine an expiry instant: leave time-to-expiry null.
    # Underlying alignment likewise needs explicitly timestamped reference observations.
    return [
        ResearchBar(
            instrument_id=json.dumps(bar.instrument_key, separators=(",", ":")),
            timestamp=bar.timestamp,
            close=float(bar.close),
            high=float(bar.high),
            low=float(bar.low),
            volume=float(bar.volume),
            open_interest=float(bar.open_interest) if bar.open_interest is not None else None,
            strike=float(bar.strike) if bar.strike is not None else None,
            provenance=manifest.provider,
            dataset_version=manifest.import_id,
            synthetic=manifest.synthetic,
        )
        for bar in bars
    ]


def build_feature_artifact(args: argparse.Namespace) -> dict[str, Any]:
    manifest, bars = read_import(args.manifest)
    inputs = research_bars(bars, manifest)
    features = build_features(inputs)
    rows = build_dataset(
        inputs,
        horizon=args.horizon,
        threshold=args.threshold,
        upper_barrier=args.upper,
        lower_barrier=args.lower,
    )
    if not rows:
        raise ValueError("not enough bars for the requested future horizon")
    target_field = "target" if args.target == "threshold" else args.target
    write_json(
        args.output,
        {
            "schema_version": "research-dataset-v1",
            "dataset_version": manifest.import_id,
            "synthetic": manifest.synthetic,
            "warning": "SYNTHETIC: engineering validation only"
            if manifest.synthetic
            else "Research only; verify source licensing, adjustments and assumptions",
            "target_field": target_field,
            "features": [row.model_dump(mode="json") for row in features],
            "rows": [row.model_dump(mode="json") for row in rows],
        },
    )
    return {
        "output": str(args.output),
        "feature_rows": len(features),
        "label_rows": len(rows),
        "synthetic": manifest.synthetic,
    }


def read_dataset(path: Path) -> tuple[list[DatasetRow], str]:
    document = json.loads(path.read_text())
    if document.get("schema_version") != "research-dataset-v1":
        raise ValueError("unsupported research dataset schema")
    rows = [DatasetRow.model_validate(row) for row in document["rows"]]
    if any(
        row.synthetic != document["synthetic"] or row.dataset_version != document["dataset_version"]
        for row in rows
    ):
        raise ValueError("dataset provenance mismatch")
    if any(a.timestamp > b.timestamp for a, b in zip(rows, rows[1:], strict=False)):
        raise ValueError("dataset rows must be chronological")
    identities = [(row.instrument_id, row.timestamp) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("dataset rows contain duplicate instrument timestamps")
    return rows, str(document["target_field"])


def train_models(args: argparse.Namespace) -> dict[str, Any]:
    from trading_agent.ml.pipeline import train_model
    from trading_agent.ml.registry import save_model
    from trading_agent.ml.split import walk_forward

    rows, target_field = read_dataset(args.dataset)
    if target_field == "future_return":
        raise ValueError(
            "future_return is a stored regression target; baseline models classify only"
        )
    splits = walk_forward(
        rows, args.train_size, args.validation_size, args.test_size, expanding=not args.rolling
    )
    if not splits:
        raise ValueError(
            "no nonempty purged folds: reduce window sizes or supply more observations"
        )
    models = []
    for split in splits:
        bundle = train_model(
            rows, split, model_kind=args.algorithm, seed=args.seed, target_field=target_field
        )
        bundle.metadata["dataset_artifact_sha256"] = checksum(args.dataset)
        path = args.registry / str(bundle.metadata["model_id"])
        save_model(bundle, path)
        models.append(str(path))
    return {"models": models, "folds": len(models), "synthetic": any(r.synthetic for r in rows)}


def evaluate_artifact(args: argparse.Namespace) -> dict[str, Any]:
    from trading_agent.ml.pipeline import evaluate_model
    from trading_agent.ml.registry import load_model

    rows, target_field = read_dataset(args.dataset)
    bundle = load_model(args.model, trusted=args.trust_local_artifact)
    period = bundle.metadata["ranges"]["test"]
    start, end = datetime.fromisoformat(period["start"]), datetime.fromisoformat(period["end"])
    if target_field != bundle.metadata["target"]["field"]:
        raise ValueError("evaluation target does not match model")
    if args.partition == "test":
        if checksum(args.dataset) != bundle.metadata.get("dataset_artifact_sha256"):
            raise ValueError(
                "saved test evaluation requires the original immutable dataset artifact"
            )
        rows = [r for r in rows if start <= r.timestamp <= end]
    else:
        cutoff = datetime.fromisoformat(bundle.metadata["ranges"]["validation"]["label_end"])
        rows = [r for r in rows if r.timestamp > cutoff]
    metrics = evaluate_model(bundle, rows, cost_bps=args.cost_bps)
    metrics.update(
        {
            "model_id": args.model.name,
            "dataset_versions": sorted({r.dataset_version for r in rows}),
            "dataset_artifact_sha256": checksum(args.dataset),
            "provenance": sorted({r.provenance for r in rows}),
            "partition": args.partition,
            "period": {
                "start": min(r.timestamp for r in rows).isoformat(),
                "end": max(r.timestamp for r in rows).isoformat(),
            },
            "synthetic": bundle.metadata["synthetic"] or any(r.synthetic for r in rows),
        }
    )
    if metrics["synthetic"]:
        metrics["warning"] = "SYNTHETIC engineering validation; no profitability evidence"
    write_json(args.output, metrics)
    return {"output": str(args.output), **metrics}


def run_backtest(args: argparse.Namespace) -> dict[str, Any]:
    from trading_agent.backtesting.engine import BacktestConfig, BacktestEngine
    from trading_agent.config.settings import Settings
    from trading_agent.data.calendar import MarketCalendar
    from trading_agent.ml.registry import load_model
    from trading_agent.research_strategies import ResearchStrategy

    manifest, bars = read_import(args.manifest)
    document = json.loads(args.config.read_text())
    settings = Settings(**document["risk"])
    values = dict(document["simulation"])
    values["provenance"] = f"{manifest.provider}; dataset={manifest.import_id}"
    calendar = MarketCalendar.model_validate_json(args.calendar.read_text())
    model = load_model(args.model, trusted=args.trust_local_artifact) if args.model else None
    model_id = "none"
    model_lineage = "not applicable"
    start = args.start
    end = args.end
    if model is not None:
        metadata = model.metadata
        cutoff = datetime.fromisoformat(metadata["ranges"]["validation"]["label_end"])
        model_id = str(metadata["model_id"])
        model_lineage = json.dumps(
            {
                "dataset_versions": metadata["dataset_versions"],
                "provenance": metadata["provenance"],
                "training_dataset_sha256": metadata["dataset_artifact_sha256"],
            },
            sort_keys=True,
        )
        if start is None:
            start = datetime.fromisoformat(metadata["ranges"]["test"]["start"])
        if end is None:
            end = datetime.fromisoformat(metadata["ranges"]["test"]["end"])
        if start <= cutoff:
            raise ValueError("model backtest start must follow the validation label horizon")
    warmup_bars = [bar for bar in bars if start is not None and bar.timestamp < start]
    if start is not None:
        bars = [bar for bar in bars if bar.timestamp >= start]
    if end is not None:
        bars = [bar for bar in bars if bar.timestamp <= end]
    if not bars:
        raise ValueError("selected backtest period contains no bars")
    values["synthetic"] = manifest.synthetic or bool(model and model.metadata["synthetic"])
    values["dataset_version"] = manifest.import_id
    values["strategy_name"] = "ml_probability_baseline" if model else "sma_crossover_demo"
    values["strategy_config"] = {
        "fast": str(args.fast),
        "slow": str(args.slow),
        "quantity": str(args.quantity),
        "model_id": model_id,
        "model_lineage": model_lineage,
        "probability_threshold": str(args.probability_threshold),
        "stop_intent_fraction": "0.05; no automatic stop execution",
        "start": start.isoformat() if start else "dataset-start",
        "end": end.isoformat() if end else "dataset-end",
    }
    config = BacktestConfig.model_validate(values)
    strategy = ResearchStrategy(
        manifest,
        fast=args.fast,
        slow=args.slow,
        quantity=args.quantity,
        model=model,
        threshold=args.probability_threshold,
    )
    result = BacktestEngine(settings, config, calendar).run(bars, strategy, warmup_bars=warmup_bars)
    result.write_reports(args.output)
    return {
        "report": str(args.output / "report.json"),
        "synthetic": config.synthetic,
        "fills": len(result.trades),
        "audit_events": len(result.audit),
    }
