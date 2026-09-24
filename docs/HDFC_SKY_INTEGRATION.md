# HDFC SKY Open API boundary — all live functionality TODO

`execution/hdfc_sky.py` is a disabled adapter. Calling `submit` always raises
`NotImplementedError`; configuration accepts only paper mode. No authentication
material, live client, endpoints or guessed request/response schemas exist.

| Future capability | Intended integration responsibility | Status |
|---|---|---|
| Authentication | Isolated official client, lifecycle/expiry handling, redacted diagnostics | TODO: official specifications |
| Order placement | Risk-approved intent mapped to documented broker request | TODO: official specifications |
| Order status | Reconcile broker IDs and partial/complete/rejected states | TODO: official specifications |
| Modify/cancel | New risk evaluation plus race-safe order state transitions | TODO: official specifications |
| Positions | Reconcile broker positions with local ledger before further execution | TODO: official specifications |
| Holdings | Separate settlement/holding model and authorized reconciliation | TODO: official specifications |

Do not fill in URLs, auth flows, field names, status enumerations or product codes
from inference. Obtain official API documentation and permitted sandbox facilities
first. Production integration also needs idempotency semantics, uncertain-result
reconciliation, durable reservations, fees and taxes, rejection mapping, transport
timeouts, audit retention and operational controls.

The agent must never receive a broker client. A future adapter cannot treat an
agent's proposed risk approval as authority. HDFC access and any credentials are
outside this task and must not be supplied to the Phase 1 application.
