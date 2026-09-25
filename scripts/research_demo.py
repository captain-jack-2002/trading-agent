"""Reproduce the full SYNTHETIC engineering workflow without network or brokerage access."""

import json
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "data" / "backtests" / f"SYNTHETIC-{uuid4().hex}"
    output.mkdir(parents=True)
    started = time.perf_counter()

    def cli(*args: str) -> dict[str, object]:
        result = subprocess.run(
            [sys.executable, "-m", "trading_agent.cli", *args],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        )
        document: dict[str, object] = json.loads(result.stdout)
        return document

    imported = cli(
        "data",
        "import",
        "examples/research/SYNTHETIC.csv",
        "--mapping",
        "examples/research/mapping.json",
        "--synthetic",
        "--root",
        str(output / "imports"),
    )
    manifest = str(imported["manifest_path"])
    dataset = str(output / "features.json")
    cli("features", "build", manifest, "--output", dataset, "--horizon", "5")
    trained = cli(
        "model",
        "train",
        dataset,
        "--registry",
        str(output / "models"),
        "--train-size",
        "120",
        "--validation-size",
        "60",
        "--test-size",
        "60",
    )
    models = trained["models"]
    assert isinstance(models, list)
    model = str(models[0])
    model_metadata = json.loads((Path(model) / "metadata.json").read_text())
    test_start = model_metadata["ranges"]["test"]["start"]
    test_end = model_metadata["ranges"]["test"]["end"]
    cli(
        "model",
        "evaluate",
        model,
        dataset,
        "--output",
        str(output / "evaluation.json"),
        "--trust-local-artifact",
    )
    for name, extra in (("sma", []), ("ml", ["--model", model, "--trust-local-artifact"])):
        cli(
            "backtest",
            "run",
            manifest,
            "--calendar",
            "examples/research/calendar.json",
            "--config",
            "examples/research/backtest.json",
            "--output",
            str(output / name),
            "--start",
            test_start,
            "--end",
            test_end,
            *extra,
        )
    print(
        json.dumps(
            {
                "warning": "SYNTHETIC engineering only; no profitability evidence",
                "artifacts": str(output.relative_to(root)),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "bars": imported["row_count"],
                "model_folds": trained["folds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
