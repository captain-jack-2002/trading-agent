# Deployment

Python 3.12 and uv are required. `uv sync --frozen` installs the checked-in lockfile.
`./scripts/check.sh` runs formatting, lint, strict source typing, tests with coverage,
and package build, keeping tool caches and temporary test files in the workspace.

Use an existing rootless Podman + Compose provider or Docker Compose installation.
`podman compose up --build -d` builds the non-root application image and starts
PostgreSQL and Valkey. App readiness requires an initialized database. Valkey failure
after startup degrades to a local TTL cache; it does not disable risk enforcement.
The provided Compose startup waits for both service health checks.

The sandbox deliberately uses passwordless PostgreSQL trust authentication only
inside an internal container network; neither data service is published. Do not
reuse it as production security configuration. The app binds to localhost on the
host, runs as UID 10001 with a read-only filesystem, dropped capabilities and a
writable temporary mount. Runtime market/broker egress is not needed.

Persistent named volumes retain the ledger and Valkey data. Ordinary
`podman compose down` retains them; do not use `down -v` unless you intend to erase
all paper history. Rebuilding/restarting the application restores portfolio and
order IDs from PostgreSQL. Initial cash applies only when creating a new ledger.

Schema creation is automatic for this initial version. Future schema changes need
versioned migrations and backup/restore checks; `create_all` does not migrate an
existing schema. Use one application worker/replica for this development release.
If database initialization fails at startup, liveness stays up and readiness/orders
return 503. Restart the app after the database is restored to retry initialization.
Never interpret a network timeout as proof an order failed: inspect `/audit` and
portfolio, and preserve the original ID when retrying.

Environment configuration is listed in `.env.example`; the application does not
implicitly read `.env`. Lists use JSON syntax. No broker secrets are accepted or
required. Settings cannot enable live mode. Calendar dates and market session
should be supplied from authorized sources for realistic simulations.

Optional PostgreSQL/Valkey integration validation instructions and actual execution
results are recorded in `OVERNIGHT_REPORT.md`. Container prerequisites that require
host privileges must be handled separately by the owner; this project does not
change the host's network, storage, systemd or security configuration.

For repeatable optional service tests, run `./scripts/check-services.sh`. It uses
`scripts/podman-workspace.sh` to keep container storage/runtime in this checkout,
starts passwordless disposable PostgreSQL/Valkey containers with `--network none`,
connects over workspace Unix sockets, tests them, and removes only the containers
it created. It needs existing working rootless Podman; it installs or changes no
host services. The short `.r` runtime directory accommodates Podman's path limit.
Image caches remain in ignored `.podman/`. This helper is for isolated validation;
use your normal Podman setup for Compose deployment.

`uv build` emits a generic warning because the uv cache is inside the checkout.
The artifact check explicitly confirms that the wheel and source distribution
exclude caches, local databases, environment files and container runtime state.
