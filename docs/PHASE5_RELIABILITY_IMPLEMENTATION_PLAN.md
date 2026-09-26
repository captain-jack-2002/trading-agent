# Phase 5A — Grounding, Hallucination Guardrails, and Model Drift Reliability

## Goal

Add a reliability layer around the current paper-only trading research stack so that:

- agent/research answers are grounded in explicit, auditable evidence;
- market/account/execution facts cannot be invented by a free-form model;
- trained models can be monitored for feature, prediction, calibration, performance, and data-quality drift;
- severe drift or data-quality failure can quarantine model-generated signals;
- retraining produces challenger candidates only and never auto-promotes a model;
- the deterministic risk engine remains the final execution authority.

This phase must not enable live brokerage execution and must not introduce profitability claims.

## Architectural boundary

```text
Trusted docs / model metadata / validated research
                    |
                    v
            Grounding / RAG layer
                    |
                    +----> Evidence packet + provenance
                    |
Typed market/account tools ----> Fact policy / hallucination guard
                    |
                    v
              Research agent
                    |
                    v
              Model proposal
                    |
          +---------+---------+
          |                   |
          v                   v
      Drift monitor       Audit trail
          |
          v
      Model health gate
          |
          v
Deterministic risk engine
          |
          v
       Paper execution
```

The risk engine must not depend on an LLM, RAG retrieval score, or free-form text.

## A. Grounding / RAG

Implement deterministic, local-first retrieval. Do not add a hosted vector database or mandatory external LLM dependency.

A first implementation may use the existing scikit-learn stack (for example TF-IDF + cosine similarity), but the abstraction must allow a future embedding retriever without changing agent contracts.

Trusted source classes:

- repository Markdown documentation;
- Phase reports and model cards;
- model-registry metadata JSON;
- validated research artifacts explicitly marked trusted;
- typed provider/tool results supplied at request time.

NSE MCP remains informational-only. It must not become training data or an executable-price source.

Add immutable typed contracts for concepts equivalent to:

- `GroundingDocument`
- `GroundingChunk`
- `SourceProvenance`
- `RetrievedEvidence`
- `GroundingRequest`
- `GroundingResult`
- `FactRequirement`
- `GroundingDecision`

Every evidence item must retain at least:

- source ID/path/tool name;
- content hash;
- source type;
- retrieval timestamp;
- source timestamp when known;
- retrieval score;
- trust class;
- informational/training/executable flags where applicable.

### Hallucination guard / fact policy

Create an explicit policy layer for facts that must never come from free-form model memory.

At minimum classify:

- executable market price;
- account balance/equity;
- open position/quantity;
- order state;
- risk limit;
- model probability/score;
- model-performance metric;
- market volume/OI/derivative fields.

Define allowed source types per fact class.

Examples:

- executable price -> executable quote provider only;
- portfolio/account facts -> authoritative ledger/broker adapter only;
- model probability -> typed model inference result only;
- model metrics -> registry/report metadata only;
- research/news interpretation -> trusted retrieval/MCP may support explanation but never execution authority.

If required evidence is absent, stale, conflicting, or below threshold, return an explicit abstention/unsupported result. Never synthesize a plausible value.

### Retrieval behavior

Implement:

- deterministic chunking;
- deterministic ranking and tie-breaking;
- configurable top-k;
- minimum retrieval score;
- source filters;
- freshness filtering for time-sensitive source types;
- duplicate-content suppression;
- no hidden network calls;
- auditable evidence packets.

Suggested CLI surface:

```bash
uv run trading-agent grounding index --source docs --output data/grounding/index
uv run trading-agent grounding query "What are the model limitations?" --index data/grounding/index
uv run trading-agent grounding inspect <evidence-id> --index data/grounding/index
```

Adapt names if needed to match existing CLI conventions.

## B. Model drift monitoring

Implement independent diagnostics for:

1. feature/data drift;
2. prediction drift;
3. calibration drift;
4. realized performance drift;
5. data-quality/schema violations.

### Training baseline

During model training, persist a monitoring baseline in model metadata using training data only.

For every selected feature, store enough deterministic summary information to compare future windows, for example:

- count/null rate;
- mean/std;
- min/max;
- quantiles;
- fixed histogram/bin edges derived from training only.

For classifiers, also persist a training-reference prediction-probability distribution.

Monitoring bins/statistics must never be fit on future validation/test/live windows.

### Drift metrics

Implement well-defined statistical measures with unit tests and deterministic behavior:

- Population Stability Index (PSI);
- Kolmogorov-Smirnov statistic where appropriate;
- Wasserstein distance;
- null-rate/schema drift;
- prediction distribution shift;
- Brier/calibration change once realized labels exist;
- precision/recall/F1 degradation once realized labels exist.

Thresholds must be configurable and versioned. Do not hard-code claims that any threshold is a universal finance-industry standard.

### Model-health state

Create typed model-health output with statuses equivalent to:

- `healthy`
- `watch`
- `quarantined`

Output must include individual diagnostics, observed values, configured limits, severity, and human-readable reasons.

A severe data-integrity violation must fail closed. A severe model-health violation must prevent that model from authorizing a signal path.

## C. Champion / challenger lifecycle

Extend the model registry with lifecycle metadata.

Support concepts equivalent to:

- `candidate`
- `challenger`
- `champion`
- `quarantined`
- `retired`

Promotion must be explicit and auditable.

No code path may do:

```text
drift -> retrain -> auto-deploy
```

Required flow:

```text
drift detected
      |
      v
candidate dataset
      |
      v
retrain challenger
      |
      v
purged walk-forward validation
      |
      v
paper/shadow validation
      |
      v
explicit promotion gate
      |
      v
champion
```

Automatic retraining may create a challenger artifact, but promotion must require an explicit promotion command or approval function.

Suggested CLI:

```bash
uv run trading-agent model health <model-id> --window <artifact>
uv run trading-agent model drift <model-id> --window <artifact>
uv run trading-agent model promote <model-id> --registry data/models
uv run trading-agent model quarantine <model-id> --reason "..."
```

Promotion/quarantine commands must create append-only audit records.

## D. Decision provenance

Create a structured decision record that can trace:

```text
market data
   -> features
   -> model + version
   -> prediction
   -> drift/model-health state
   -> retrieved evidence
   -> fact-policy decisions
   -> agent proposal
   -> risk decision
   -> paper order / rejection
```

The record should contain stable IDs and hashes rather than copying arbitrary large source text.

At minimum record:

- instrument;
- event timestamp;
- dataset/import version;
- feature version;
- model ID/version;
- model prediction and probability where relevant;
- model-health status;
- drift diagnostics reference;
- evidence IDs and source hashes;
- unsupported/abstained claims;
- proposed action;
- deterministic risk decision;
- final paper-execution outcome.

Do not store secrets or credentials in decision records.

## E. Fail-safe behavior

The system must fail closed for trading-relevant uncertainty.

Examples:

- missing executable quote -> no execution;
- stale authoritative quote -> no execution;
- invalid feature schema -> model rejected;
- quarantined model -> model signal rejected;
- ungrounded model metric claim -> abstain;
- unavailable RAG index -> research answer may degrade to explicit "insufficient evidence", never fabricated facts;
- NSE MCP unavailable -> research context degrades safely; execution path unaffected.

Do not create a fallback where the LLM invents a missing signal.

## F. Integration constraints

Preserve all current Phase 1–4 boundaries:

- paper execution only;
- HDFC SKY live adapter disabled;
- NSE MCP informational-only;
- NSE MCP not training eligible;
- NSE MCP not executable-price eligible;
- deterministic risk engine remains sole execution authority;
- serialized joblib artifacts still require explicit local trust;
- no credentials in repository, tests, logs, fixtures, or prompts.

Use existing project style: Python 3.12, Pydantic immutable contracts where appropriate, typing, Ruff, mypy, pytest, uv.

Avoid large new infrastructure dependencies unless justified.

## G. Tests

Add comprehensive tests for:

### Grounding

- deterministic chunking/ranking;
- source hashing/provenance;
- source trust flags;
- top-k/min-score behavior;
- stale evidence rejection;
- duplicate evidence suppression;
- conflicting source handling;
- abstention when evidence is absent;
- fact-policy source restrictions;
- executable price cannot come from RAG/MCP/free-form text.

### Drift

- identical distributions produce near-zero drift;
- shifted distributions trigger expected drift;
- PSI edge cases and zero bins;
- KS/Wasserstein behavior;
- missing columns/null-rate drift;
- prediction distribution drift;
- calibration/performance drift with realized labels;
- thresholds are configuration-driven;
- model health transitions: healthy -> watch -> quarantined.

### Lifecycle

- challenger creation;
- champion remains unchanged after retraining;
- promotion is explicit;
- quarantine blocks model use;
- audit entries are append-only;
- invalid/corrupt registry state fails closed.

### End-to-end

At least one test must prove:

```text
model signal + healthy model + valid typed quote
-> risk engine still has final authority
```

and another must prove:

```text
high-confidence model prediction + quarantined model
-> no paper order
```

## H. Documentation

Add/update:

- `docs/GROUNDING.md`
- `docs/MODEL_DRIFT.md`
- `docs/MODEL_LIFECYCLE.md`
- `docs/ARCHITECTURE.md`
- `docs/ML_PIPELINE.md`
- `docs/MODEL_CARD.md`
- README phase/roadmap section
- `PHASE5_RELIABILITY_REPORT.md`

The Phase 5 report must explain what was implemented, exact commands, tests, limitations, remaining risks, and explicitly state that no real-market performance conclusion can be drawn from synthetic validation.

## I. Quality gates

Before completion run:

```bash
uv sync --frozen
./scripts/check.sh
uv run ruff check .
uv run mypy src
uv build
git status --short
```

All existing tests must continue to pass. New functionality should have high coverage and must not reduce the existing safety boundaries.

Do not commit generated model binaries, local indexes, runtime databases, credentials, or ignored research outputs.

## J. Git workflow

Work only on branch:

```text
phase5-grounding-drift
```

Make logically separated commits.

Do not merge to `main` and do not tag a release automatically. Leave the branch ready for review.

## K. Completion notification

When the complete implementation run finishes, send a notification to:

```text
https://ntfy.sh/trading-agent-rag-drift-implementation
```

On success:

```bash
curl -fsS \
  -H "Title: Trading Agent Phase 5A complete" \
  -H "Priority: high" \
  -d "Grounding/RAG, hallucination guardrails, drift monitoring, champion-challenger lifecycle, tests and documentation completed successfully." \
  https://ntfy.sh/trading-agent-rag-drift-implementation
```

On failure:

```bash
curl -fsS \
  -H "Title: Trading Agent Phase 5A failed" \
  -H "Priority: urgent" \
  -d "Phase 5A implementation did not complete successfully. Review Codex output and quality-gate failures." \
  https://ntfy.sh/trading-agent-rag-drift-implementation
```

Notification failure must not mask the implementation/test exit status.

## Definition of done

Phase 5A is complete only when:

- local RAG/grounding is implemented and auditable;
- trading-critical fact classes have explicit authoritative-source policies;
- unsupported facts cause abstention instead of fabrication;
- model training persists drift baselines from training-only data;
- feature/prediction/calibration/performance drift can be evaluated;
- model health can quarantine model-generated signals;
- champion/challenger lifecycle exists with explicit promotion only;
- decision provenance is persisted or emitted in a structured auditable form;
- risk engine remains final authority;
- all quality gates pass;
- documentation and `PHASE5_RELIABILITY_REPORT.md` are complete;
- execution remains paper-only.
