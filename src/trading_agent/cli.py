"""Explicit local-only research commands; no brokerage or credential access."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Paper-only historical research pipeline")
    groups = root.add_subparsers(dest="group", required=True)
    data = groups.add_parser("data", help="Inspect, validate and import supplied files")
    actions = data.add_subparsers(dest="action", required=True)
    inspect = actions.add_parser("inspect", help="Display CSV headers and up to five rows")
    inspect.add_argument("source", type=Path)
    for name in ("validate", "import", "normalize", "build-parquet"):
        command = actions.add_parser(name)
        command.add_argument("source", type=Path)
        command.add_argument("--mapping", required=True, type=Path)
        if name != "validate":
            command.add_argument("--root", type=Path, default=Path("data/datasets"))
            command.add_argument("--synthetic", action="store_true")
    summary = actions.add_parser("summarize")
    summary.add_argument("manifest", type=Path)
    features = groups.add_parser("features", help="Build trailing features and future labels")
    feature_actions = features.add_subparsers(dest="action", required=True)
    build = feature_actions.add_parser("build")
    build.add_argument("manifest", type=Path)
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--horizon", type=int, default=5)
    build.add_argument(
        "--target",
        choices=("direction", "future_return", "threshold", "barrier"),
        default="direction",
    )
    build.add_argument("--threshold", type=float, default=0.0)
    build.add_argument("--upper", type=float, default=0.02)
    build.add_argument("--lower", type=float, default=0.01)
    backtest = groups.add_parser("backtest", help="Risk-gated next-bar simulation")
    backtest_actions = backtest.add_subparsers(dest="action", required=True)
    run = backtest_actions.add_parser("run")
    run.add_argument("manifest", type=Path)
    run.add_argument("--calendar", required=True, type=Path)
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--start", type=datetime.fromisoformat, help="inclusive closed-bar timestamp")
    run.add_argument("--end", type=datetime.fromisoformat, help="inclusive closed-bar timestamp")
    run.add_argument("--fast", type=int, default=5)
    run.add_argument("--slow", type=int, default=20)
    run.add_argument("--quantity", type=int, default=1)
    run.add_argument("--model", type=Path)
    run.add_argument("--trust-local-artifact", action="store_true")
    run.add_argument("--probability-threshold", type=float, default=0.5)
    model = groups.add_parser("model", help="Chronological CPU baseline training/evaluation")
    model_actions = model.add_subparsers(dest="action", required=True)
    train = model_actions.add_parser("train")
    train.add_argument("dataset", type=Path)
    train.add_argument("--registry", type=Path, default=Path("data/models"))
    train.add_argument("--algorithm", choices=("logistic", "random_forest"), default="logistic")
    train.add_argument("--train-size", type=int, required=True)
    train.add_argument("--validation-size", type=int, required=True)
    train.add_argument("--test-size", type=int, required=True)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--rolling", action="store_true")
    evaluate = model_actions.add_parser("evaluate")
    evaluate.add_argument("model", type=Path)
    evaluate.add_argument("dataset", type=Path)
    evaluate.add_argument("--output", required=True, type=Path)
    evaluate.add_argument(
        "--trust-local-artifact",
        action="store_true",
        help="Allow loading an executable serialized model you trust",
    )
    evaluate.add_argument("--partition", choices=("test", "out-of-sample"), default="test")
    evaluate.add_argument("--cost-bps", type=float, default=0)
    return root


def dispatch(args: argparse.Namespace) -> dict[str, Any]:
    if args.group == "data" and args.action == "inspect":
        with args.source.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            rows = []
            for _, row in zip(range(5), reader, strict=False):
                rows.append(row)
            return {"file": str(args.source), "columns": reader.fieldnames, "sample_rows": rows}
    if args.group == "data":
        from trading_agent.data.providers import CSVMapping, CSVProvider
        from trading_agent.data.storage import import_csv, verify_import

        if args.action == "summarize":
            manifest = verify_import(args.manifest.parent)
            return manifest.model_dump(mode="json")
        mapping = CSVMapping.model_validate_json(args.mapping.read_text())
        if args.action == "validate":
            count = 0
            instruments: set[tuple[str, ...]] = set()
            for bar in CSVProvider(mapping).iter_bars(args.source):
                count += 1
                instruments.add(bar.instrument_key)
            if not count:
                raise ValueError("source contains no data rows")
            return {"validation": "passed", "rows": count, "instruments": sorted(instruments)}
        manifest = import_csv(args.source, mapping, args.root, synthetic=args.synthetic)
        return {
            **manifest.model_dump(mode="json"),
            "manifest_path": str(args.root / manifest.import_id / "manifest.json"),
        }
    from trading_agent.cli_research import (
        build_feature_artifact,
        evaluate_artifact,
        run_backtest,
        train_models,
    )

    if args.group == "features":
        return build_feature_artifact(args)
    if args.group == "model":
        return train_models(args) if args.action == "train" else evaluate_artifact(args)
    return run_backtest(args)


def main(argv: list[str] | None = None) -> int:
    command = parser()
    args = command.parse_args(argv)
    try:
        output = dispatch(args)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        command.exit(2, f"error: {exc}\n")
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
