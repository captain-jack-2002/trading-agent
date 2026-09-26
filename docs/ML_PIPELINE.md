# Offline CPU learning pipeline

`ml.dataset.build_dataset(bars, horizon=5, threshold=0, upper_barrier=.02,
lower_barrier=.01)` returns chronological `DatasetRow` records extending FeatureRow.
Every record includes label_end, horizon, label_version, feature_version, provenance,
dataset_version and synthetic. The last horizon bars per instrument have no target
and are omitted. Future return is close[t+h]/close[t]-1; direction is strictly >0;
target is strictly greater than threshold. Barriers inspect future high/low in order.
If both barriers touch in the same first hit bar, status is ambiguous: intrabar
ordering is unknowable. All label_end values retain the full configured horizon.

`walk_forward(rows, train_size, validation_size, test_size, step=None,
expanding=True)` counts distinct timestamps, keeping all instruments at one timestamp
together. It never shuffles. Rolling mode fixes training window length. Train labels
ending at or after validation start and validation labels ending at or after test
start are purged. Empty folds are omitted. Repeated test windows can overlap if
step is smaller than test_size; do not aggregate them as independent observations.

`train_model(rows, split, model_kind='logistic', seed=42, target_field='target')`
supports LogisticRegression and single-worker RandomForestClassifier CPU baselines.
Target fields are target, direction, or barrier (upper=1, lower=0; ambiguous/neither
excluded). Future-return regression is intentionally unsupported. Median imputation,
scaling and feature selection fit training only. Entirely missing training features
are excluded; features missing only in later data use training medians. Both training
classes are required. Validation/test metrics do not select hyperparameters.

`predict_probabilities(bundle, rows)` accepts FeatureRow records and checks version,
required feature keys and finite values. It returns positive-class probabilities.
Caller must enforce deployment chronology; the function itself is a pure predictor.
`evaluate_model` accepts labelled rows. Training cannot establish economic usefulness.

`save_model(bundle, Path)` creates a new immutable registry folder with model.joblib
and metadata.json, including SHA256, features, library versions, seed, hyperparameters,
label definitions, ranges, provenance and metrics. `load_model(path, trusted=True)`
requires explicit trust because joblib can execute code. A checksum detects accidental
corruption, not malicious artifacts or a replaced checksum. Never trust a downloaded
model merely because it passes a hash. Registry writes refuse existing directories.
Repeat training under matching package versions produces matching predictions; creation
timestamps mean serialized bytes are not guaranteed identical. All example fixtures
are SYNTHETIC engineering data, never evidence of trading performance.

Registry publication stages both files in a sibling temporary directory and renames
only after successful serialization; cooperating publishers use an exclusive lock.
Failures clean staging files and leave no published model. Metadata includes a UUID
model_id. Do not edit artifacts manually or publish through unrelated concurrent
writers. Training rejects duplicate timestamps, nonchronological rows/indices,
boolean/noninteger indices, horizon overlaps and mixed target definitions. Input
rows must share dataset version/provenance/synthetic status; assemble mixed sources
into an explicitly versioned dataset before training. Label horizon and binary labels
use strict integers and synthetic status uses a strict boolean at JSON boundaries.


## Phase 5A monitoring and lifecycle

Training also persists `monitoring_baseline` from the selected training rows and
training predictions only. It contains exact compressed distributions, training
histograms and null/statistical/calibration/performance references. A shifted
validation/test partition cannot change it. Missing legacy baselines fail closed
for monitored signal use. Full methodology and policy configuration are in
[MODEL_DRIFT.md](MODEL_DRIFT.md).

New publications register as challengers. `LifecycleRegistry` verifies immutable
artifact identity against append-only events, and explicit promotion requires a
reviewed purged walk-forward and paper/shadow validation attestation. Retraining
never changes the champion. Quarantine blocks loaded model signal paths and cannot
be cleared by a healthy monitoring window. See [MODEL_LIFECYCLE.md](MODEL_LIFECYCLE.md).

`ModelPaperExecutor` accepts fresh FeatureRow windows and order intent, computes its
own prediction and current health, checks lifecycle under a lock through paper
submission, rechecks feature freshness after inference/quote latency, and records
structured provenance. It checks every input timestamp is after validation label
horizons. Required research evidence may be supplied as a
GroundingRequest; unsupported evidence abstains. Quotes/ledger/risk are independently
obtained by the broker. Prediction/evaluation functions remain pure offline tools,
including retrospective diagnosis of quarantined models; they never place orders.
Retrospective backtests also gate fills on causal-window model health without
changing deployed lifecycle. Supervised metrics and all synthetic tests remain
engineering validation only.
