# Phase 4 model training implementation plan

**Goal:** reproducible SYNTHETIC engineering validation of supervised models through
the existing causal-feature, registry, risk and simulated-execution infrastructure.

**Architecture:** extend `ml/dataset.py`, `ml/split.py`, `ml/pipeline.py` and the local
registry; add focused audit, metrics and experiment modules. Extend the existing
CLI. Use the existing 300-bar `examples/research/SYNTHETIC.csv` without generating
or importing market observations. Fixed settings, no return-driven tuning.

**Constraints:** work in the requested checkout on `phase4-model-training`; retain
all historical tags/history; execution stays paper; no broker/MCP training input;
generated files remain under ignored data paths. User's detailed Phase 4 request
is the authoritative specification and authorizes incremental execution.

## Milestones

- [x] 1. Labels and leakage: tests first for inclusive threshold, barrier ambiguity
  and full-horizon purge, duplicate identities, timestamp groups, feature schema,
  synthetic/MCP policy, chronological holdout, rolling/expanding folds. Fix labels
  with a target-version bump. Add causal distances/range through existing features.
- [x] 2. Models: tests first for deterministic logistic/random forest/LightGBM CPU
  classifiers and clean regression support; train-only imputation/scaling,
  probability distributions/calibration, dependency/Python/Git/dataset metadata,
  serialization/checksums and malformed/missing input. No parameter search.
- [x] 3. Workflow: extend model train/evaluate/list/show and backtest run; persist
  configs, predictions, fold/aggregate metrics, audit and metadata. Add fixed cash,
  trailing-direction and SMA comparisons using the same test windows and existing
  risk/cost/slippage engines. Test CLI, denied trades and strict OOS boundaries.
- [x] 4. Evidence: run a reproducible script using only the existing fixture,
  evaluate direction/threshold/barrier/regression, holdout and both walk-forward
  modes; write `PHASE4_MODEL_TRAINING_REPORT.md` with exact metrics, fold ranges,
  IDs, limitations, artifact paths, runtime and recommendations for real-data
  evaluation only. Document commands and dataset licensing handoff for Phase 5.
- [x] 5. Review and release checks: fresh whole-branch review; fix significant
  findings with regression tests; run `./scripts/check.sh`, explicitly run Ruff
  format/check, mypy, `uv build`, and rootless Podman build if dependencies change.
  Commit stable milestones; verify clean tree, expected branch and unchanged tags.
  Send the requested ntfy notification as the final external action.

## Validation and review focus

Run focused tests after each change and the full suite at stable milestones.
Review adversarial feature columns, contaminated provenance, equal timestamps
across instruments, overlapping label intervals and fold tests, empty/single-class
partitions, boundary probabilities, shifted evaluation artifacts, and warmup bars.
Barrier ties are unknowable from OHLC and excluded from supervised metrics;
unresolved barriers are negative (target not reached), with full-horizon purging.
Existing targets-v1 semantics remain readable but Phase 4 emits targets-v2.
Walk-forward aggregate metrics describe repeated model fits, not independent market
evidence; final holdout remains separate from all walk-forward training/labels.

## Progress

Initial verification: requested directory/branch; clean tree. Tag object IDs:
`v0.1.0-phase1=6b4ac627c9386bf249ae992809b794588fd3e2eb`,
`v0.2.0-phase2=9c51a651004e0472c8cf0f7ac077767213835132`,
`v0.3.0-phase3-nse-mcp=6b6ba60031d22bf1afc0b152ed9c3ec65c82c655`.

Milestones 1–2: 184 tests passed, two optional service tests skipped; mypy and
focused Ruff passed. Added LightGBM 4.7.0 (MIT) with deterministic one-thread CPU
settings and container libgomp1. New tests first demonstrated missing policies,
metadata, inclusive thresholds and regression support, then passed after changes.

Milestone 3: 191 tests passed, two optional service tests skipped; Ruff and mypy
passed. The full CLI flow, registry inspection, immutable run artifacts, source
rebuild and risk-denial integration pass. Container built and imported LightGBM
4.7.0 with paper settings. Barrier metric filtering now preserves evaluation
calendar boundaries; a failing regression test demonstrated and pins the fix.

Independent whole-branch review found four Important issues, all reproduced first
by nine failing tests in `test_phase4_review.py`: timestamp-range reevaluation
reintroduced purged irregular-instrument samples; OOS evaluation skipped source
rebuild; substring provenance accepted a non-synthetic feed; permissive parsing
silently discarded source deny controls. Fixes persist exact partition identities,
rebuild OOS inputs, narrow synthetic inference, and reject unknown bar/row fields.
The 52-test focused suite passed; full suite passed 200 tests with two optional
service skips. Separate isolated service checks passed both tests. Two additional
causality/F&O regression checks passed with the 13-test leakage suite.

Review scope rulings: no market usefulness or investment suitability claim is
made; cryptographic provenance is outside this local checksum system and remains
an explicit limitation. Final report and release gates are verified by the primary
agent after the read-only review. No Critical issues or deferred Minor findings.
First fixed-config evidence run completed 84 models/105 backtests; final evidence
will be regenerated from the committed review fixes without changing parameters.

Final policy check additionally reproduced an inherited standalone-backtest path
that accepted CSV bars labeled as MCP. The CLI now applies the same synthetic/MCP
policy before simulation (regression test RED→GREEN). No MCP responses were used;
the test relabels the existing synthetic CSV. Coverage subprocess temporary files
are now ignored (`.coverage.*`) so runtime state cannot contaminate Git provenance.
The complete gate passed 203 tests with two separately verified service tests;
Ruff, mypy, package/artifact validation and the updated rootless image build passed.
Two full runs produced exactly equal 84 model and 105 backtest metric sets. Final
report evidence is rerun after the ignore fix to eliminate transient dirty flags.

Final evidence: `data/models/PHASE4-SYNTHETIC-63d2625cfabd48a2877b08f98ae00978`.
84 model fits, 105 backtests, all 465 runtime artifact checksums verified; exact
partition prediction membership and shared backtest windows verified. All model
metadata records clean source commit af5d3be. Runtime 174.785 s; recorded fit/eval
8.338 s; peak process RSS 223.46 MiB. Metrics identical across three full runs.
Final gates: 203 tests passed (+2 service tests passed separately), Ruff format/lint,
mypy, uv build, distribution checks, rootless image build and isolated import smoke
all passed. Report contains the complete metrics, registry IDs, limitations and
Phase 5 plan. Final documentation commit and clean-tree/tag verification precede
the authorized ntfy completion notice.
