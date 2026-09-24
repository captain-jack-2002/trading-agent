#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .tmp
export UV_CACHE_DIR="$PWD/.uv-cache"
export TMPDIR="$PWD/.tmp"
uv sync --frozen
uv run ruff format --check src tests scripts
uv run ruff check src tests scripts
uv run mypy src
uv run pytest --cov=trading_agent --cov-report=term-missing
uv build
uv run python scripts/check_artifacts.py
