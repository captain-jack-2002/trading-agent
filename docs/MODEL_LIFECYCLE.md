# Explicit model lifecycle

Model artifacts remain immutable (`model.joblib`, `metadata.json`). Lifecycle
state is held separately in a local SQLite journal, `REGISTRY/lifecycle.db`.
Each transition records sequence, artifact UUID, metadata digest, previous/event
hash, UTC timestamp, actor and reason. SQL triggers reject event updates/deletes;
replay verifies schema/controls, transition rules, journal head, artifact identity,
checksums and a single champion. Invalid, missing or corrupt state rejects signals.
Local filesystem/DB administrator access remains a trust boundary: a hash chain is
not external signing or protection against a malicious administrator rewriting all
state. Retain independent backups/audit exports for operational assurance.

```text
candidate -> challenger -> explicit promotion -> champion -> retired
     |           |                                  |
     +-----------+----------------------------------+-> quarantined
```

Publication registers a new model as challenger. Candidate enrollment is explicit
for legacy artifacts; candidates cannot authorize signals until challenged.
Challengers may generate paper/shadow proposals after health checks. They never
replace the champion through training, drift detection, or a healthy report.
Quarantined/retired models cannot authorize proposals. Quarantine is latched;
retraining publishes a new challenger. There is no automatic recovery/promotion.

```bash
uv run trading-agent model enroll OLD_MODEL --actor operator --reason "Verified local artifact"
uv run trading-agent model challenge OLD_MODEL --actor operator --reason "Reviewed candidate"
uv run trading-agent model promote MODEL --walk-forward walk.json --paper-validation paper.json --approved-by operator --reason "Reviewed validation"
uv run trading-agent model quarantine MODEL --reason "Severe drift"
uv run trading-agent model retire MODEL --reason "Withdrawn"
uv run trading-agent model audit --registry data/models
```

Every model command accepts `--registry`; mutation commands retain operator/reason
in the append-only journal. Promotion verifies both local validation artifact
hashes and their binding to artifact UUID and binary SHA256. Walk-forward JSON
requires `kind: "purged_walk_forward"`, `passed: true`, `purged: true`; paper JSON
requires `kind: "paper_shadow_validation"`, `passed: true`, `paper_only: true`.
Both require `model_id` and `model_sha256`. These are reviewed attestations; boolean
flags do not calculate or statistically certify a validation experiment. Produce
and inspect actual supporting results before approving. The Python approval
function also accepts reviewed digest attestations without paths for embedding in
an external approval workflow. It is always an explicit operator action.

An explicit promotion atomically retires the previous champion and appends the
new champion record. Failed evidence/identity checks roll back all transitions.
The journal lock is held across model-backed paper submission, so a concurrent
quarantine cannot race the final signal eligibility check. The ledger transaction
still independently obtains quotes, checks deterministic risk and commits fills.

Pure retrospective `predict_probabilities`, `predict_returns`, and model evaluation
remain available for trusted offline analysis, including quarantined models. They
are not execution interfaces. Loaded research strategies and model-backed paper
execution consult current lifecycle state. Legacy artifacts without monitoring
baselines require retraining for monitored signal use. HDFC SKY stays disabled;
NSE MCP remains informational-only. Synthetic validation is engineering only.
