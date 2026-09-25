"""Run the fixed Phase 4 SYNTHETIC experiment; no downloads, tuning or live access."""

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from trading_agent.cli import dispatch, parser
from trading_agent.cli_research import read_dataset
from trading_agent.data.storage import checksum
from trading_agent.features.research import ResearchBar
from trading_agent.ml.audit import SYNTHETIC_WARNING
from trading_agent.ml.dataset import build_dataset
from trading_agent.ml.pipeline import predict_probabilities, predict_returns, train_model
from trading_agent.ml.registry import load_model
from trading_agent.ml.split import chronological_split
from trading_agent.research_io import write_json

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples/research"


def cli(*args: object) -> dict[str, Any]:
    return dispatch(parser().parse_args([str(arg) for arg in args]))


def table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        if value is None:
            return "undefined"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("|", "/")

    return (
        "\n".join(
            [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join("---" for _ in headers) + " |",
                *("| " + " | ".join(cell(v) for v in row) + " |" for row in rows),
            ]
        )
        + "\n"
    )


def render_report(summary: dict[str, Any], output: Path) -> str:
    models, backtests = summary["models"], summary["backtests"]
    sample = models[0]["metadata"]
    text = [
        "# Phase 4 prediction-model training and evaluation",
        "\n**SYNTHETIC ONLY — engineering validation. These invented prices provide no "
        "evidence of expected market performance. No model is production-ready or suitable "
        "for live trading. execution_mode remains paper.**\n",
        "## Dataset and reproducibility\n",
        f"Exact source: `examples/research/SYNTHETIC.csv`; SHA-256 `{summary['source_sha256']}`. "
        "300 one-minute DEMO equity bars, one fixture session on 2024-01-02 "
        "09:16–14:15 Asia/Kolkata. Prices are the pre-existing trigonometric fixture. "
        "No NSE MCP data, downloads, brokerage connections or real orders were used. "
        "295 labeled samples per objective; 300 feature rows; one instrument.\n",
        f"Dataset version/import ID: `{summary['dataset_version']}`. "
        f"Feature version `{sample['feature_version']}`; target version `targets-v2`; "
        f"model version `{sample['model_version']}`. Seed 42; fixed model parameters and "
        "thresholds, no search or return-driven tuning.\n",
        f"Training Git commit `{sample['git_commit']}`; dirty checkout: `{sample['git_dirty']}`. "
        f"Python `{sample['python_version']}`. Dependencies: "
        + ", ".join(f"{k}={v}" for k, v in sample["dependency_versions"].items())
        + ".\n",
        "Reproduce from this checkout with `uv sync --frozen` and "
        "`uv run python scripts/phase4_training.py --report PHASE4_MODEL_TRAINING_REPORT.md`. "
        "Each invocation creates a new ignored run; UUIDs, creation times, binary checksums "
        "and timings differ. Deterministic predictions/metrics are checked within 1e-12 "
        "tolerance for repeated holdout fits. Lockfile and source/config checksums are recorded.\n",
        "## Targets and features\n",
        "All labels use bars t+1 through t+5, with t close as the known reference. Direction: "
        "close[t+5]/close[t]-1 > 0. Threshold: the same return >= 0.002. Regression: that "
        "continuous return. Barrier: +0.004 high reached before -0.004 low within five bars; "
        "upper first=1, lower first or neither=0. A bar touching both is ambiguous and excluded "
        "from supervised fitting/metrics, but receives predictions during simulation. No "
        "intrabar ordering is invented. Every label retains the full horizon for purging. "
        "Legacy targets-v1 remains readable; new inclusive threshold/barrier semantics are v2.\n",
        "Causal features used ("
        + str(len(sample["features"]))
        + "): `"
        + "`, `".join(sample["features"])
        + "`.\n",
        "Unavailable/all-null training columns are omitted: `"
        + "`, `".join(sample["preprocessing"]["excluded_features"])
        + "`. "
        "F&O calculations remain supported and tested instrument-locally, but this equity "
        "fixture has no OI, aligned underlying, strike or expiry observations. No F&O "
        "values were fabricated. Warmup nulls are imputed using training medians only.\n",
        "## Models and preprocessing\n",
        "LogisticRegression (max_iter=2000); RandomForest (100 trees, depth 6, minimum "
        "leaf 2, n_jobs=1); LightGBM (80 trees, depth 4, 15 leaves, minimum child 10, "
        "deterministic CPU, force_col_wise, n_jobs=1). Regression uses Ridge(alpha=1), "
        "RandomForestRegressor and LGBMRegressor. sklearn Pipeline persists median "
        "imputation and StandardScaler fitted exclusively on each training partition. "
        "No transformer, calibration model or parameter search is fitted to validation/test. "
        "All-null column selection also uses training only.\n",
        "LightGBM's [MIT license](https://github.com/lightgbm-org/LightGBM/blob/main/LICENSE) "
        "and [CPU settings](https://lightgbm.readthedocs.io/en/latest/Parameters.html) "
        "were checked. Runtime container includes libgomp1; no GPU dependency added.\n",
        "## Chronology and leakage audit\n",
        "Final holdout uses timestamp groups: 150 train, 60 validation, 85 test before "
        "purging. The last five training/validation labels are purged, leaving 145/55/85 "
        "(barrier ambiguity can further reduce fitting/metric counts). Walk-forward uses "
        "only the 205 development labels ending strictly before holdout start. Initial "
        "windows 70/30/30, step 30, three folds each for rolling and expanding. Earlier "
        "fold test tails are purged against the next fold test start. All instruments at "
        "a timestamp remain on the same side of every boundary.\n",
        "Audit checks: source feature/label rebuild; prefix invariance; future perturbation; "
        "feature allowlist; duplicate instrument timestamps/samples; global time boundaries; "
        "disjoint indices; strict label_end < next partition start; train-only imputer/scaler "
        "statistics; cross-fold test label separation; no holdout label data in development. "
        "Repeated numeric patterns at different timestamps are legitimate observations, "
        "not duplicate sample identities. Same instruments recur across time; these tests "
        "do not claim transfer to unseen instruments. Overlapping labels within training "
        "are permitted; no cross-boundary overlap is permitted.\n",
        "Phase 3 MCP flags remain informational_only=true, training_eligible=false, "
        "executable_price=false. Phase 4 rejects MCP provenance and non-synthetic inputs. "
        "Provenance and checksums are local engineering controls, not cryptographic proof "
        "of the origin of a deliberately relabeled dataset.\n",
    ]
    ranges = []
    for model in models:
        if model["target"] != "direction" or model["algorithm"] != "logistic":
            continue
        for name, period in model["metadata"]["ranges"].items():
            ranges.append(
                [
                    model["mode"],
                    model["fold"],
                    name,
                    period["count"],
                    period["start"],
                    period["end"],
                    period["label_end"],
                ]
            )
    text.append(table(["Mode", "Fold", "Partition", "Rows", "Start", "End", "Label end"], ranges))
    text.append("\n## Per-model and per-fold supervised test metrics (SYNTHETIC)\n")
    text.append(
        "PR-AUC is average precision; undefined ROC/PR for single-class partitions "
        "is null. Calibration uses five fixed [0,.2,…,1] bins; the final bin includes 1. "
        "ECE is count-weighted absolute prediction/observation discrepancy.\n"
    )
    classification, regression = [], []
    for model in models:
        metrics = model["metadata"]["metrics"]["test"]
        identity = [model["target"], model["mode"], model["fold"], model["algorithm"]]
        if model["target"] == "future_return":
            regression.append(
                identity
                + [metrics[k] for k in ("count", "mae", "rmse", "r2", "directional_accuracy")]
            )
        else:
            classification.append(
                identity
                + [
                    metrics[k]
                    for k in (
                        "count",
                        "precision",
                        "recall",
                        "f1",
                        "roc_auc",
                        "pr_auc",
                        "brier",
                        "expected_calibration_error",
                        "confusion_matrix",
                        "class_distribution",
                    )
                ]
            )
    text.append(
        table(
            [
                "Target",
                "Mode",
                "Fold",
                "Algorithm",
                "N",
                "Precision",
                "Recall",
                "F1",
                "ROC-AUC",
                "PR-AUC",
                "Brier",
                "ECE",
                "CM [[TN,FP],[FN,TP]]",
                "Classes",
            ],
            classification,
        )
    )
    text.append(
        table(
            [
                "Target",
                "Mode",
                "Fold",
                "Algorithm",
                "N",
                "MAE",
                "RMSE",
                "R²",
                "Directional accuracy",
            ],
            regression,
        )
    )
    text.append("\n## Walk-forward aggregates and calibration\n")
    text.append(
        table(
            ["Target", "Mode", "Algorithm", "Metric", "Mean", "Std", "Min", "Max"],
            [
                [
                    run["target"],
                    run["mode"],
                    algorithm,
                    metric,
                    values["mean"],
                    values["std"],
                    values["min"],
                    values["max"],
                ]
                for run in summary["runs"]
                if run["mode"] != "holdout"
                for algorithm, aggregates in run["aggregates"].items()
                for metric, values in aggregates.items()
                if metric in ("brier", "f1", "rmse", "expected_calibration_error")
            ],
        )
    )
    text.append("\nHoldout probability distributions and calibration bins:\n")
    text.append(
        table(
            [
                "Target",
                "Algorithm",
                "Mean",
                "Std",
                "Min",
                "Max",
                "Bin counts",
                "Calibration (count, predicted, observed)",
            ],
            [
                [
                    m["target"],
                    m["algorithm"],
                    *[
                        m["metadata"]["metrics"]["test"]["probability_distribution"][k]
                        for k in ("mean", "std", "min", "max", "counts")
                    ],
                    [
                        (b["count"], round(b["predicted"], 5), round(b["observed"], 5))
                        for b in m["metadata"]["metrics"]["test"]["calibration"]
                    ],
                ]
                for m in models
                if m["mode"] == "holdout" and m["target"] != "future_return"
            ],
        )
    )
    text.append("\n## Risk-gated backtests and baseline comparisons (SYNTHETIC)\n")
    text.append(
        "Historical bars → causal features → predictions → proposed signals → existing "
        "RiskEngine → next-bar simulated fill → portfolio. Classifiers use p>=0.5; "
        "regressors use predicted return>0; one share, long-only, 5% stop intent. "
        "Cash emits nothing; naive follows the previous one-bar close direction; "
        "SMA uses existing 5/20 crossover. All methods share calendar, warmup, test "
        "window, capital, risk, cost and slippage settings. Warmup bars never trade. "
        "No automatic barrier/stop execution; proposed direction may differ from "
        "positions after rejection. Final holdings are marked, not force-liquidated.\n"
    )
    text.append(
        "Initial capital INR 100000; original risk defaults. Illustrative existing "
        "equity-delivery cost fixture: brokerage 0.0001/side (cap INR 7), STT sell "
        "0.0003, exchange 0.00002/side, SEBI 0.000001/side, stamp buy 0.00001, "
        "GST 10% on brokerage+exchange+SEBI. These are invented test rates. Slippage "
        "2 bps/side plus adverse 0.01 tick rounding; volatility multiplier=0. "
        "Turnover is notional traded / initial capital; exposure is average INR "
        "notional including initial zero. Winners/losers are net closed sell-fill "
        "PnLs. Transaction fees and slippage are separately recorded.\n"
    )
    holdout_bt = [b for b in backtests if b["mode"] == "holdout"]
    text.append(
        table(
            ["Method", *[b["name"] for b in holdout_bt]],
            [
                [metric, *[b["metrics"].get(metric) for b in holdout_bt]]
                for metric in holdout_bt[0]["metrics"]
            ],
        )
    )
    text.append(
        "\nPer-fold backtest results (complete risk audits, fills, equity, costs and "
        "all remaining ratios are in each report.json):\n"
    )
    text.append(
        table(
            [
                "Mode",
                "Fold",
                "Method",
                "Return",
                "Drawdown",
                "Fills",
                "Turnover",
                "Fees",
                "Slippage",
            ],
            [
                [
                    b["mode"],
                    b["fold"],
                    b["name"],
                    *[
                        b["metrics"][k]
                        for k in (
                            "total_return",
                            "max_drawdown",
                            "fill_count",
                            "turnover",
                            "costs",
                            "slippage",
                        )
                    ],
                ]
                for b in backtests
                if b["mode"] != "holdout"
            ],
        )
    )
    text.append(
        "\nAnnualized return and Calmar are undefined because every window is shorter "
        "than one year (94500 minute periods). Sharpe, Sortino and annualized volatility "
        "are mechanical diagnostics on a tiny synthetic sample, without inferential "
        "meaning. Fold portfolios reset; overlapping rolling/expanding runs must not "
        "be pooled into a portfolio curve.\n"
    )
    text.append("\n## Ranking for FUTURE REAL-DATA EVALUATION only\n")
    text.append(
        "The following priority is based on SYNTHETIC data only. Eligibility requires "
        "passed leakage and reproducibility checks. Fixed ranking key: worst across "
        "rolling/expanding (mean Brier + fold Brier std), then holdout Brier, holdout "
        "ECE, holdout turnover, runtime. This emphasizes calibration and stability; "
        "returns and accuracy are not ranking inputs.\n"
    )
    text.append(
        table(
            [
                "Rank",
                "Classifier",
                "WF mean+std (worst mode)",
                "Holdout Brier",
                "Holdout ECE",
                "Turnover",
                "Fit seconds",
            ],
            summary["ranking"],
        )
    )
    text.append(
        "\nCarry all three modest CPU classifiers into licensed-data evaluation, with "
        f"`{summary['ranking'][0][1]}` first in this synthetic engineering ordering. "
        "Retain Logistic Regression as the interpretable reference regardless of rank. "
        "Retain Ridge as the regression reference and compare tree regressors on "
        "licensed data. Single-session scores cannot establish market robustness.\n"
    )
    text.append("\n## Registry, artifacts and runtime\n")
    text.append(
        f"Run root: `{output.relative_to(ROOT)}`. Summary: `summary.json`; source/config "
        "checksums: `checksums.json`; imported immutable bars: `imports/`; datasets: "
        "`datasets/`; models: `registry/<model_id>/{model.joblib,metadata.json}`; each "
        "`registry/runs/SYNTHETIC-*/` has training_config, metrics, walk_forward, "
        "predictions, leakage_audit and checksums JSON. Simulations: "
        "`backtests/<mode>/<fold>/<method>/report.json` and report.md. "
        "All runtime artifacts are ignored; no model binaries or generated datasets "
        "are committed. UUIDs identify local artifacts, not stable model contents.\n"
    )
    text.append(
        table(
            ["Target", "Mode", "Fold", "Algorithm", "Model ID", "Fit seconds"],
            [
                [
                    m["target"],
                    m["mode"],
                    m["fold"],
                    m["algorithm"],
                    m["metadata"]["model_id"],
                    m["metadata"]["training_runtime_seconds"],
                ]
                for m in models
            ],
        )
    )
    text.append(
        f"\nWall runtime: {summary['runtime_seconds']:.3f} seconds. Summed recorded model "
        f"fit/evaluation time: {summary['training_seconds']:.3f} seconds. "
        f"Process peak RSS: {summary['peak_rss_mib']:.2f} MiB (Linux ru_maxrss, process-wide; "
        "not per-model allocation). These timings describe this environment only. "
        f"Repeated holdout prediction checks: {summary['deterministic_checks']} passed.\n"
    )
    text.append("\n## Limitations and Phase 5\n")
    text.append(
        "This fixture has one artificial instrument/session, smooth invented patterns, "
        "no corporate actions, liquidity regimes or real spreads; apparent prediction "
        "quality is not evidence of a tradeable edge. Calibration diagnostics are "
        "small-sample estimates; no recalibration is claimed. Single-class/constant-target "
        "undefined metrics stay null. OHLC barrier ties are unknowable. Existing "
        "simulation lacks automatic stops, market impact, derivatives margin and "
        "exchange matching. Risk rejection remains authoritative.\n"
    )
    text.append(
        "Once authorized/licensed NSE historical data is supplied, first add explicit "
        "license/provenance approval and a reviewed ingestion policy; do not disable "
        "the MCP boundary. Verify sessions, timezone/availability timestamps, corporate "
        "actions, symbol/contract lifetimes, survivorship, F&O underlying alignment, "
        "and realistic exchange/broker costs. Freeze a multi-regime final holdout, "
        "repeat purged rolling/expanding evaluation and instrument-held-out tests, "
        "add uncertainty estimates and benchmark comparisons. Fit any calibrator only "
        "on permitted development observations. Phase 5 should establish those real-data "
        "contracts, monitoring/drift and offline replay acceptance criteria while keeping "
        "execution paper. No live brokerage integration is recommended by this phase.\n"
    )
    return "\n".join(text)


def main() -> None:
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--report", type=Path, help="Write the requested human-readable report")
    args = arguments.parse_args()
    started = time.perf_counter()
    output = ROOT / "data/models" / f"PHASE4-SYNTHETIC-{uuid4().hex}"
    imported = cli(
        "data",
        "import",
        FIXTURE / "SYNTHETIC.csv",
        "--mapping",
        FIXTURE / "mapping.json",
        "--root",
        output / "imports",
        "--synthetic",
    )
    manifest = imported["manifest_path"]
    simulation = json.loads((ROOT / "examples/backtest/illustrative_costs.json").read_text())
    simulation["slippage"]["bps"] = "2"
    config_path = output / "backtest_config.json"
    write_json(config_path, {"risk": {"execution_mode": "paper"}, "simulation": simulation})
    models, runs, backtests = [], [], []
    deterministic_checks = 0
    for target in ("direction", "threshold", "barrier", "future_return"):
        full = output / "datasets" / f"{target}.json"
        cli(
            "features",
            "build",
            manifest,
            "--output",
            full,
            "--target",
            target,
            "--horizon",
            5,
            "--threshold",
            0.002,
            "--upper",
            0.004,
            "--lower",
            0.004,
        )
        document = json.loads(full.read_text())
        # Development labels may use only bars strictly before the final test start.
        cutoff = document["rows"][210]["timestamp"]
        source = [ResearchBar.model_validate(b) for b in document["source_bars"]]
        source = [bar for bar in source if bar.timestamp.isoformat() < cutoff]
        dev = output / "datasets" / f"{target}-development.json"
        write_json(
            dev,
            {
                **document,
                "source_bars": [b.model_dump(mode="json") for b in source],
                "rows": [
                    r.model_dump(mode="json")
                    for r in build_dataset(
                        source, horizon=5, threshold=0.002, upper_barrier=0.004, lower_barrier=0.004
                    )
                ],
            },
        )
        for mode in ("holdout", "expanding", "rolling"):
            dataset = full if mode == "holdout" else dev
            train, validation, test = (150, 60, 85) if mode == "holdout" else (70, 30, 30)
            extra = (
                ["--holdout"] if mode == "holdout" else ["--rolling"] if mode == "rolling" else []
            )
            trained = cli(
                "model",
                "train",
                dataset,
                "--registry",
                output / "registry",
                "--algorithm",
                "all",
                "--train-size",
                train,
                "--validation-size",
                validation,
                "--test-size",
                test,
                *extra,
            )
            metrics = json.loads((Path(trained["artifacts"]) / "metrics.json").read_text())
            runs.append(
                {
                    "target": target,
                    "mode": mode,
                    "artifacts": trained["artifacts"],
                    "aggregates": metrics["aggregate"],
                }
            )
            for record, path in zip(metrics["models"], trained["models"], strict=True):
                metadata = json.loads((Path(path) / "metadata.json").read_text())
                models.append(
                    {
                        "target": target,
                        "mode": mode,
                        "algorithm": record["algorithm"],
                        "fold": record["fold"],
                        "metadata": metadata,
                    }
                )
                if mode == "holdout":
                    rows, field = read_dataset(dataset)
                    split = chronological_split(rows, 150, 60)
                    repeated = train_model(
                        rows, split, model_kind=record["algorithm"], target_field=field
                    )
                    saved = load_model(Path(path), trusted=True)
                    selected = [rows[i] for i in split.test]
                    predict = (
                        predict_returns if target == "future_return" else predict_probabilities
                    )
                    assert all(
                        abs(a - b) <= 1e-12
                        for a, b in zip(
                            predict(repeated, selected), predict(saved, selected), strict=True
                        )
                    )
                    deterministic_checks += 1
                name = f"{target}-{record['algorithm']}"
                period = metadata["ranges"]["test"]
                methods = [(name, ["--model", path, "--trust-local-artifact"])]
                if target == "direction" and record["algorithm"] == "logistic":
                    methods += [
                        (baseline, ["--baseline", baseline])
                        for baseline in ("cash", "naive", "sma")
                    ]
                for method, flags in methods:
                    location = output / "backtests" / mode / str(record["fold"]) / method
                    cli(
                        "backtest",
                        "run",
                        manifest,
                        "--calendar",
                        FIXTURE / "calendar.json",
                        "--config",
                        config_path,
                        "--output",
                        location,
                        "--start",
                        period["start"],
                        "--end",
                        period["end"],
                        *flags,
                    )
                    report = json.loads((location / "report.json").read_text())
                    backtests.append(
                        {
                            "mode": mode,
                            "fold": record["fold"],
                            "name": method,
                            "metrics": report["metrics"],
                            "path": str(location),
                        }
                    )
            print(
                f"SYNTHETIC {target}/{mode}: {len(metrics['models'])} models and backtests",
                flush=True,
            )
    ranking = []
    for algorithm in ("logistic", "random_forest", "lightgbm"):
        wf = [
            r["aggregates"][algorithm]["brier"]
            for r in runs
            if r["target"] == "direction" and r["mode"] != "holdout"
        ]
        model = next(
            m
            for m in models
            if m["target"] == "direction" and m["mode"] == "holdout" and m["algorithm"] == algorithm
        )
        metrics = model["metadata"]["metrics"]["test"]
        bt = next(
            b for b in backtests if b["mode"] == "holdout" and b["name"] == f"direction-{algorithm}"
        )
        ranking.append(
            [
                algorithm,
                max(v["mean"] + v["std"] for v in wf),
                metrics["brier"],
                metrics["expected_calibration_error"],
                bt["metrics"]["turnover"],
                model["metadata"]["training_runtime_seconds"],
            ]
        )
    ranking.sort(key=lambda r: r[1:])
    summary = {
        "warning": SYNTHETIC_WARNING,
        "synthetic": True,
        "models": models,
        "runs": runs,
        "backtests": backtests,
        "source_sha256": checksum(FIXTURE / "SYNTHETIC.csv"),
        "dataset_version": imported["import_id"],
        "ranking": [[i, *r] for i, r in enumerate(ranking, 1)],
        "runtime_seconds": time.perf_counter() - started,
        "training_seconds": sum(m["metadata"]["training_runtime_seconds"] for m in models),
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "deterministic_checks": deterministic_checks,
    }
    write_json(output / "summary.json", summary)
    write_json(
        output / "checksums.json",
        {
            "warning": SYNTHETIC_WARNING,
            "sha256": {
                str(p.relative_to(output)): checksum(p)
                for p in sorted(output.rglob("*"))
                if p.is_file()
            },
            "source_files": {
                str(p.relative_to(ROOT)): checksum(p)
                for p in (
                    FIXTURE / "SYNTHETIC.csv",
                    FIXTURE / "mapping.json",
                    FIXTURE / "calendar.json",
                    ROOT / "uv.lock",
                    ROOT / "scripts/phase4_training.py",
                )
            },
        },
    )
    if args.report:
        args.report.write_text(render_report(summary, output))
    print(
        json.dumps(
            {
                "artifacts": str(output),
                "models": len(models),
                "backtests": len(backtests),
                "runtime_seconds": summary["runtime_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
