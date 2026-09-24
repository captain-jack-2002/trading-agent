#!/usr/bin/env bash
# Optional isolated validation helper: keep Podman storage/runtime in this checkout.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .tmp .podman/storage .podman/config .podman/cache .podman/data .r
export TMPDIR="$PWD/.tmp"
export XDG_CONFIG_HOME="$PWD/.podman/config"
export XDG_CACHE_HOME="$PWD/.podman/cache"
export XDG_DATA_HOME="$PWD/.podman/data"
export XDG_RUNTIME_DIR="$PWD/.r"
exec podman --root "$PWD/.podman/storage" --runroot "$PWD/.r" "$@"
