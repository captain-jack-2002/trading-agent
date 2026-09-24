#!/usr/bin/env bash
# Real PostgreSQL/Valkey checks with no exposed ports, network or credentials.
set -euo pipefail
cd "$(dirname "$0")/.."
runner="$PWD/scripts/podman-workspace.sh"
service_suffix="$$"
pg_name="trading-check-pg-$service_suffix"
vk_name="trading-check-vk-$service_suffix"
socket_root="$PWD/.tmp/services-$service_suffix"
mkdir -p "$socket_root/pg" "$socket_root/valkey"
chmod 777 "$socket_root/pg" "$socket_root/valkey"
cleanup() {
  "$runner" rm -f "$pg_name" "$vk_name" >/dev/null 2>&1 || true
}
trap cleanup EXIT
"$runner" run -d --name "$pg_name" --network none \
  -e POSTGRES_HOST_AUTH_METHOD=trust -e POSTGRES_DB=paper -e POSTGRES_USER=paper \
  -v "$socket_root/pg:/var/run/postgresql" docker.io/library/postgres:17-alpine
"$runner" run -d --name "$vk_name" --network none --user 0 \
  -v "$socket_root/valkey:/sockets" docker.io/valkey/valkey:8-alpine \
  valkey-server --port 0 --unixsocket /sockets/valkey.sock --unixsocketperm 777 \
  --save '' --appendonly no
for attempt in {1..30}; do
  if "$runner" exec "$pg_name" pg_isready -U paper -d paper >/dev/null \
    && "$runner" exec "$vk_name" valkey-cli -s /sockets/valkey.sock ping >/dev/null; then
    break
  fi
  sleep 1
done
export UV_CACHE_DIR="$PWD/.uv-cache"
export TMPDIR="$PWD/.tmp"
export TRADING_TEST_POSTGRES_SOCKET="$socket_root/pg"
export TRADING_TEST_VALKEY_SOCKET="$socket_root/valkey/valkey.sock"
uv run pytest tests/test_services.py -q
