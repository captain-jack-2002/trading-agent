# Security and safety scope

Phase 1 has no live execution implementation and no API route to enable one.
The HDFC stub raises unconditionally. The agent cannot call execution; the paper
adapter always runs risk policy against a locked authoritative ledger. Caches
cannot authorize an order. All state changes and audit effects commit atomically.

No real keys, credentials or secrets belong in this repository. `.env*` is ignored
except the placeholder example. Do not give broker credentials to this application.
Logging selects event/status/ID fields and avoids dumping settings or exception
connection URLs. Audit events contain paper order intent and results; treat future
real trading records as sensitive if the project scope changes.

The API has no authentication or multi-user isolation and is for localhost-only
use. The Compose network is internal, data ports are unpublished, and the app port
binds to 127.0.0.1. Passwordless database trust is strictly a local paper-sandbox
convenience; it does not protect against a malicious host/container administrator.
Do not expose the stack to untrusted clients. Production authentication, TLS,
service identities and operational hardening require a separate reviewed phase.

Supply-chain dependencies are open-source and locked with hashes in `uv.lock`.
Container image tags are versioned but not digest-pinned; verify and pin approved
image digests before a reproducible deployment. No system privilege changes,
privileged containers, host mounts, real brokers or financial transactions are
needed. Request validation rejects nonfinite prices, invalid quantities and naive
timestamps. The official market calendar and complete exchange semantics remain
integration work; paper results cannot establish live safety or profitability.
