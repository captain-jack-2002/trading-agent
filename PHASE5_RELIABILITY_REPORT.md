# Phase 5A reliability implementation report

**Synthetic engineering validation only.** Nothing in this phase establishes
expected trading performance, profitability, real-market accuracy or live suitability.
The authoritative scope is `docs/PHASE5_RELIABILITY_IMPLEMENTATION_PLAN.md`.
Work remains on `phase5-grounding-drift`; no merge, release, tag or live execution.

## Architecture

Local trusted documents and typed provider/tool adapters produce immutable evidence
packets. TF-IDF retrieval supplies cited research; an independent authoritative-fact
policy returns supported, unsupported, stale, conflicting or below-threshold decisions.
Unsupported research abstains. Free-form text supplies no trading-critical facts.

Training stores monitoring baselines from the selected training partition only.
Drift evaluates features, prediction distributions, calibration, realized performance
and schema/null integrity. Severe violations produce typed quarantined health.
Model-backed paper submission computes its own predictions, checks feature chronology
and freshness, and latches severe drift in an append-only lifecycle journal. The
current lifecycle lock spans the final broker call. The broker independently obtains
validated quotes and the locked ledger, rechecks feature freshness after inference/
provider latency, evaluates the unchanged deterministic risk
engine, and atomically persists fill/rejection, ledger and decision provenance.

Model artifacts remain immutable. Publication and lifecycle enrollment, including
failure cleanup, share a lock. State is replayed from hash-chained SQLite events.
Explicit promotion retires an old champion and installs a reviewed challenger in
one lifecycle transaction. Neither monitoring nor retraining promotes/deploys.
Quarantine is latched; new training creates a distinct challenger.

Retrospective model backtests attach causal-window health and reject unhealthy fills
alongside deterministic risk; they do not mutate deployment lifecycle. Pure prediction
and evaluation remain offline diagnostic tools, including quarantined artifact analysis.
Manual paper order intent remains compatible. Provenance is nested in the existing
`paper_order` event, preserving existing API event counts; model attempts also emit
`model_decision`. Pre-order abstention persists a decision without creating an order.

## Added modules and integration files

- `grounding/contracts.py`: immutable documents, chunks, provenance, requests,
  requirements, evidence, fact observations, decisions and packets.
- `grounding/adapters.py`: separate typed quote, ledger, order, inference, metric,
  risk-settings, market-field and informational MCP adapters; exact-payload binding.
- `grounding/retrieval.py`: deterministic local retriever protocol, TF-IDF retrieval,
  atomically persisted JSON indexes, integrity checks and explicit-root loaders.
- `grounding/service.py` and `grounding/__init__.py`: fact allowlist, freshness,
  conflict checks, abstention and public contracts.
- `ml/drift.py`: training-reference distributions, PSI/KS/Wasserstein, quality,
  prediction/calibration/performance diagnostics and immutable health/thresholds.
- `ml/lifecycle.py`: candidate/challenger/champion/quarantined/retired state,
  locked append-only journal, replay integrity and explicit promotion.
- `agent/reliability.py`: independent inference, current health/lifecycle checks,
  optional research grounding and risk-gated paper execution.
- `models/decision.py`: compact immutable decision provenance.
- `cli_reliability.py`: local grounding, monitoring and audited lifecycle commands.
- Updated `ml/pipeline.py`, `ml/registry.py`, `agent/orchestrator.py`,
  `execution/paper.py`, `research_strategies.py`, `backtesting/engine.py`, `cli.py`
  and `.gitignore` for training, execution, research, CLI and local artifacts.
- Added `docs/GROUNDING.md`, `docs/MODEL_DRIFT.md`, `docs/MODEL_LIFECYCLE.md`;
  updated architecture, ML pipeline, model card, NSE MCP, risk documentation and README.

The unfinished inherited working tree had overlapping grounding contracts and tests.
Those drafts were consolidated around one public contract and execution path;
preexisting Phase 1–4 tests retain their original behavior. No dependencies were added.

## CLI commands

```bash
uv run trading-agent grounding index --source docs --output data/grounding/index
uv run trading-agent grounding query "What are the model limitations?" --index data/grounding/index
uv run trading-agent grounding inspect EVIDENCE_ID --index data/grounding/index
uv run trading-agent model health MODEL --registry data/models
uv run trading-agent model drift MODEL --window window.json --registry data/models --trust-local-artifact
uv run trading-agent model health MODEL --window window.json --thresholds policy.json --trust-local-artifact
uv run trading-agent model enroll MODEL --registry data/models --actor operator --reason "Verified local artifact"
uv run trading-agent model challenge MODEL --registry data/models --actor operator --reason "Reviewed candidate"
uv run trading-agent model promote MODEL --registry data/models --walk-forward walk.json --paper-validation paper.json --approved-by operator --reason "Reviewed validation"
uv run trading-agent model quarantine MODEL --registry data/models --reason "Severe drift"
uv run trading-agent model retire MODEL --registry data/models --reason "Withdrawn"
uv run trading-agent model audit --registry data/models
```

Queries support top-k, minimum score, source-type/ID filters, freshness and aware as-of
time. Indexing supports explicit trusted validated research artifacts and registry JSON.
Missing/corrupt indexes explicitly abstain. Health without a window reports lifecycle
and states that health was not evaluated. Monitoring never accepts caller predictions:
it independently infers with explicit local joblib trust. Severe schema/baseline failures
can quarantine without deserialization. Monitoring hashes one byte snapshot used for
both features and labels, stores immutable reports, and detects changed report content.
Reliability CLI errors do not echo potentially sensitive rejected payloads.

## RAG methodology

Fixed word chunks (180 default), normalized word TF-IDF and cosine similarity,
ordered source/ordinal ties, configurable top-k/minimum score, content deduplication,
source filters, and timestamp freshness. Source/chunk SHA256 identities and evidence
packets retain trust/eligibility flags and timestamps. Future embedding retrievers
can implement the same protocol. Local JSON indexes are atomically replaced and
recomputed/verified when loaded; no vector database, network fallback or LLM is required.

All trading-critical fact classes require their allowed typed adapters. Labeling text
as a quote does not confer authority; changed copies and JSON-restored adapter packets
cannot retain authoritative observations. Numeric values are normalized before conflict
checks. Conflicts are evaluated before top-k can hide any source. Executable-price
facts require typed executable quotes; account facts require ledger snapshots; risk
limits use whitelisted settings; probabilities use inference; metrics use typed registry/
report metadata; volume/OI/derivative facts use separate market-field adapters. Known MCP
lineage is rejected even if explicitly cast into quote or market-field DTOs. MCP evidence
always remains informational-only, training-ineligible and executable-price-ineligible.

## Drift methodology

Each selected training feature and training prediction distribution stores counts/null
rates, population mean/std, extrema, percentiles, training decile cut points, histogram
counts and exact compressed empirical values/counts. Selected features and references
are not fit or updated on validation, test or monitoring windows. Classifier baselines
store training Brier/ECE and precision/recall/F1; regressors store MAE/RMSE/directional
accuracy. These in-sample references are diagnostic and can be optimistic.

PSI uses fixed training bins with additive proportion smoothing (1e-6), avoiding zero-bin
infinities and sample-size artifacts. KS reports the maximum empirical CDF difference;
no significance or independence claim is made. One-dimensional Wasserstein integrates
absolute CDF differences, reporting raw units and gating by training-std normalization
(unit scale for constant references). Schema, unexpected columns, feature version,
null rates, nonfinite data, identities, chronology and baseline corruption fail closed.
Prediction distributions use the same metrics. Once aligned labels are realized at the
evaluation timestamp, calibration/performance compare Brier/ECE increases and precision/
recall/F1 declines; regression errors/directional accuracy are also monitored.

`DriftThresholds` is immutable, configurable and versioned. Defaults and formulas are
listed in `docs/MODEL_DRIFT.md`; they are engineering policy, not an industry standard.
Small windows produce watch. Missing prediction/label diagnostics are explicitly pending;
healthy applies to evaluated diagnostics only. A healthy report cannot clear quarantine.

## Lifecycle behavior

New artifacts are challengers; legacy artifacts require explicit enrollment and candidate
advancement. Current state, artifact identity, checksums, hash chain, SQL append-only
controls and journal head are verified before use. Quarantined and retired artifacts
cannot authorize signals, including a saved original ModelBundle or one loaded before
quarantine. Publication rollback removes its artifact while enrollment remains locked.

Promotion is explicitly approved and audited. The CLI verifies local purged walk-forward
and paper/shadow manifests, content hashes, UUID/binary-hash binding and required flags.
The Python approval function also supports operator-approved digest attestations. These
attestations are not automated statistical certification: operators must review the
actual supporting experiments. Failed promotion rolls back champion replacement.
Training or drift detection has no automatic deployment/promotion path.

## Tests and review

Added suites: `test_grounding.py`, `test_grounding_adapters.py`,
`test_grounding_integrity.py`, `test_grounding_policy.py`, `test_drift.py`,
`test_model_lifecycle.py`, `test_phase5_cli.py`, `test_phase5_execution.py`,
`test_reliability_cli.py`, `test_reliability_execution.py`,
`test_reliability_integration.py`.

Tests cover deterministic ranking/hash/dedup/filter/freshness behavior, source restrictions,
conflicts/abstention, evidence tampering, identical/shifted distributions, PSI zero-bin
and sample-size behavior, exact KS/Wasserstein examples, constant/null/schema windows,
training-only leakage, prediction/calibration/performance shifts, realized-label horizons,
corrupt baselines/metadata, configurable states, append-only/corrupt lifecycle behavior,
publication/enrollment/cleanup concurrency, explicit promotion and invalid attestations,
loaded/saved model quarantine, CLI round trips and monitoring snapshot/report integrity,
and slow inference with a fresh new quote but expired features.

Required safety proofs:

- A: healthy random forest, measured probability >0.8, valid quote: capital-limit rejection
  still comes from deterministic risk; an eligible intent still receives risk approval.
- B: that high-confidence model quarantined before submission creates neither order nor fill.
- C: missing/invalid quotes are rejected; RAG/MCP/free-form output cannot replace them.
- D: retraining publishes a challenger while the existing champion remains unchanged;
  only a separate explicit reviewed promotion changes champion.
- E: MCP evidence remains informational/training-ineligible/price-ineligible; MCP feature
  lineage and known MCP cast quote/market-field lineage cannot authorize execution.

An independent read-only reviewer checked safety and completeness. Its publication race,
saved-original bundle quarantine bypass and monitoring snapshot mismatch were reproduced
with failing tests and fixed. Additional regressions cover final feature freshness
after inference/provider latency, failed-publication cleanup, severe schema quarantine,
invalid report content and redacted CLI errors. The final diff
review checks leakage, unsafe fills, network dependencies, secrets, lifecycle mutation,
risk bypasses, artifact packaging and failure paths. No live server/broker call is needed
for ordinary tests. Exact final gate results and coverage follow below.

## Limitations and remaining risks

- Synthetic-only engineering validation; licensed real-data evaluation remains a later phase.
- TF-IDF measures lexical relevance, with no semantic understanding or generated narrative.
  Trusted document selection and trusted typed adapter producers remain operator boundaries.
- Feature provenance/schema cannot prove causal correctness of arbitrarily forged input rows;
  the causal import/rebuild pipeline and trusted caller must remain the source of observations.
- Empirical reference metadata grows with unique training observations. Local retrieval fits
  TF-IDF per query and is intended for repository-scale corpora, not massive hosted search.
- Default drift thresholds/window sizes need domain validation. Serial dependence, repeated
  tests and regime changes limit interpretation. Training reference metrics may overstate
  performance. Realized diagnostics remain pending until true outcomes are supplied.
- Watch permits paper proposals with diagnostics; healthy never promises future reliability.
- Lifecycle uses POSIX file locks and local SQLite. Use a controlled local filesystem and
  cooperating writers. External administrators can rewrite files/state; hashes are not
  signatures. Backups/external audit retention are operational responsibilities.
- Artifact files and lifecycle SQLite are separate resources. A process crash at publication
  can leave an unregistered artifact; it fails closed until reviewed enrollment. Paper ledger
  and lifecycle are separate databases; the lock prevents cooperating quarantine races but
  does not implement distributed transaction recovery or a live broker state machine.
- Reviewed promotion manifests/digest attestations are operator approvals, not independent
  experiment certification. No automatic retraining scheduler or automatic deployment exists.
- Joblib is executable and still requires explicit trust. Checksums do not make untrusted
  downloaded binaries safe. No runtime model binaries/indexes/databases are committed.
- HDFC SKY live execution is disabled. NSE MCP is informational-only. Deterministic risk
  remains final execution authority. There are no mandatory cloud/LLM dependencies.

## Exact quality-gate results

All requested gates passed on the implemented source. The final full covered run
collected 327 tests: **325 passed, 2 skipped, 0 failures**, in **251.22 seconds**.
The two existing opt-in service tests skipped because isolated PostgreSQL/Valkey
socket environment variables were not supplied. No ordinary test contacts NSE,
a cloud/LLM service or a broker.

| Required command | Exit status | Exact result |
| --- | --- | --- |
| `uv sync --frozen` | 0 | `Checked 63 packages in 0.70ms` |
| `./scripts/check.sh` | 0 | Frozen sync; 110 files already formatted; Ruff clean; mypy clean; 325 passed, 2 skipped; 92% coverage; sdist/wheel built; artifacts verified |
| `uv run ruff check .` | 0 | `All checks passed!` |
| `uv run mypy src` | 0 | `Success: no issues found in 72 source files` |
| `uv build` | 0 | Built `dist/trading_agent-0.1.0.tar.gz` and `dist/trading_agent-0.1.0-py3-none-any.whl` |
| `git status --short` | 0 | Empty output after final documentation commit; branch `phase5-grounding-drift` |

`./scripts/check.sh` also verified all **72 Python modules** in the wheel and no
local runtime state in wheel/sdist. Its uv build emitted the existing non-fatal
warning that `.uv-cache` lies inside the source directory; artifact verification
confirmed that cache/runtime state was excluded. `git diff --check` passed. Secret
signature scanning found no private-key/token signatures in changed files. The
risk engine, HDFC SKY adapter, NSE MCP client, dependency manifest and lockfile
are unchanged from the Phase 1–4 safety boundary.

### Coverage

Coverage is statement coverage from the final complete `pytest --cov=trading_agent
--cov-report=term-missing` run inside `scripts/check.sh`: **3914 statements, 294
missed, 92% overall** (previous Phase 4 report: 91%). It is not a proof of correctness
or full branch coverage.

| Reliability module | Coverage |
| --- | ---: |
| Agent grounding/orchestrator | 100% |
| Model paper executor | 96% |
| Grounding adapters | 100% |
| Grounding contracts | 97% |
| Local retriever/loaders | 96% |
| Fact policy/service | 98% |
| Drift diagnostics | 92% |
| Lifecycle | 92% |
| Reliability CLI | 92% |
| Paper execution adapter | 98% |
| Decision provenance contract | 100% |

Synthetic tests establish software behavior and guardrails only. They provide no
expected trading-performance conclusion. The branch is committed and ready for
review; it was not merged or released.
