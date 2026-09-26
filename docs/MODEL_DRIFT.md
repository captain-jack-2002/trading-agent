# Model drift monitoring

`ml.drift` compares an aligned observation window with immutable training-only
references saved in `metadata.json` by `train_model`. Baselines are computed after
target-specific training filtering, using only selected training features and
training predictions. Validation, test and monitoring data never fit bins or
baseline statistics. References include count/null rate, mean/population standard
deviation, extrema, 5/25/50/75/95 percentiles, training decile cut points with
open-ended tails, histogram counts and compressed exact empirical distributions.
Unique values plus counts preserve exact KS and Wasserstein comparisons without
expanding repeated samples. Metadata grows with unique training observations.

| Diagnostic | Definition |
| --- | --- |
| PSI | Sum `(q-p)*log(q/p)` on fixed aligned training bins; add 1e-6 to proportions and renormalize to handle empty bins independently of sample size |
| KS | Maximum difference between empirical CDFs; statistic only, no p-value or independence assertion |
| Wasserstein | Integral of absolute empirical CDF difference; raw units reported and divided by training standard deviation for gating (unit scale for constant references) |
| Null drift | Absolute change in per-feature null proportion |
| Schema/integrity | Required columns/version, finite numeric values, aligned lengths, chronology, identities and valid baseline summaries |
| Prediction drift | Same distribution diagnostics on probabilities or regression scores |
| Calibration | Increase in Brier and fixed probability-bin expected calibration error |
| Performance | Decline in precision/recall/F1, or regression directional accuracy; increase in MAE/RMSE |

Training metrics are in-sample reference diagnostics and may be optimistic. They
are not acceptance evidence or expected trading returns. Calibration/performance
are evaluated only with realized, identity-aligned outcomes whose label horizon
has ended at evaluation time. Bare label arrays require DatasetRow horizons and
must match the stored target. `RealizedLabel` also supports explicit external
outcome receipts. Missing labels or predictions are reported as not evaluated;
they are never fabricated. A healthy result applies only to evaluated diagnostics.

`DriftThresholds` is frozen and versioned. Defaults are illustrative engineering
policy: PSI .1/.25, KS .15/.3, normalized Wasserstein .5/1, null .1/.3,
Brier/ECE increase .05/.15, performance decline .1/.25 and regression error
increase .1/.5 (watch/quarantine). Minimum sample count is 20; smaller windows
produce watch. These are not universal finance standards. Tune and review policy
for licensed data, serial dependence, instruments and window size. A custom JSON
policy uses the same field names, `version`, and `{watch, quarantine}` limits.

Health is `healthy`, `watch` or `quarantined`, with per-diagnostic observations,
limits, severity and reasons. Any severe integrity violation or threshold breach
quarantines. Watch is usable for paper proposals with the diagnostic retained.
Quarantine persists in the lifecycle journal; a later healthy window cannot clear
it. Monitoring never retrains or promotes a model.

```bash
uv run trading-agent model health MODEL --registry data/models
uv run trading-agent model drift MODEL --window window.json --registry data/models --trust-local-artifact
uv run trading-agent model health MODEL --window window.json --thresholds policy.json --trust-local-artifact
```

A window is a local JSON object with `schema_version: "monitoring-window-v1"`,
aware `evaluated_at`, and `rows` of FeatureRow/DatasetRow objects. Optional `labels`
are RealizedLabel objects (`instrument_id`, `timestamp`, `label_end`, `value`).
Predictions are independently inferred; supplied prediction fields are rejected.
Healthy feature windows require explicit joblib trust to evaluate prediction drift.
Schema/baseline failure can quarantine without deserializing an executable artifact.
Reports are immutable JSON under `REGISTRY/monitoring`, addressed by their digest.
Repeated identical reports are idempotent. Evaluations never unquarantine or deploy.

All synthetic tests are engineering validation, not expected trading performance.
