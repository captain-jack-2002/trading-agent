# Phase 4 prediction-model training and evaluation

**SYNTHETIC ONLY — engineering validation. These invented prices provide no evidence of expected market performance. No model is production-ready or suitable for live trading. execution_mode remains paper.**

## Dataset and reproducibility

Exact source: `examples/research/SYNTHETIC.csv`; SHA-256 `4d8928a38f603c721d1e09ab3129cf8aae656541fe5c3b09c3f8d043eb9154fd`. 300 one-minute DEMO equity bars, one fixture session on 2024-01-02 09:16–14:15 Asia/Kolkata. Prices are the pre-existing trigonometric fixture. No NSE MCP data, downloads, brokerage connections or real orders were used. 295 labeled samples per objective; 300 feature rows; one instrument.

Dataset version/import ID: `8623274ce161bcfb7eeaaa577decbfdfa01021eef77a77dea13f3f93fc8d470d`. Feature version `research-v2`; target version `targets-v2`; model version `phase4-v1`. Seed 42; fixed model parameters and thresholds, no search or return-driven tuning.

Training Git commit `af5d3beff0efda0721dbfbc7857fca6ed95cd07b`; dirty checkout: `False`. Python `3.12.3`. Dependencies: numpy=2.5.3, scipy=1.18.1, scikit-learn=1.9.1, lightgbm=4.7.0, joblib=1.6.0, pydantic=2.13.5, pyarrow=23.0.1.

Reproduce from this checkout with `uv sync --frozen` and `uv run python scripts/phase4_training.py --report PHASE4_MODEL_TRAINING_REPORT.md`. Each invocation creates a new ignored run; UUIDs, creation times, binary checksums and timings differ. Deterministic predictions/metrics are checked within 1e-12 tolerance for repeated holdout fits. Lockfile and source/config checksums are recorded.

## Targets and features

All labels use bars t+1 through t+5, with t close as the known reference. Direction: close[t+5]/close[t]-1 > 0. Threshold: the same return >= 0.002. Regression: that continuous return. Barrier: +0.004 high reached before -0.004 low within five bars; upper first=1, lower first or neither=0. A bar touching both is ambiguous and excluded from supervised fitting/metrics, but receives predictions during simulation. No intrabar ordering is invented. Every label retains the full horizon for purging. Legacy targets-v1 remains readable; new inclusive threshold/barrier semantics are v2.

Causal features used (41): `atr_14`, `atr_pct`, `bollinger_lower`, `bollinger_position`, `bollinger_upper`, `ema_12`, `ema_26`, `ema_distance_12`, `ema_distance_26`, `high_low_volatility_20`, `log_return_1`, `log_return_10`, `log_return_20`, `log_return_5`, `ma_distance_20`, `macd`, `macd_histogram`, `macd_signal`, `momentum_20`, `realized_volatility_20`, `return_1`, `return_10`, `return_20`, `return_5`, `return_acceleration`, `rolling_high_20`, `rolling_low_20`, `rolling_range_pct_20`, `rsi_14`, `sma_10`, `sma_20`, `sma_5`, `sma_50`, `sma_distance_10`, `sma_distance_20`, `sma_distance_5`, `sma_distance_50`, `volume_acceleration`, `volume_ratio_20`, `volume_zscore_20`, `zscore_20`.

Unavailable/all-null training columns are omitted: `basis`, `moneyness`, `oi_change`, `open_interest`, `price_oi_relationship`, `time_to_expiry_days`. F&O calculations remain supported and tested instrument-locally, but this equity fixture has no OI, aligned underlying, strike or expiry observations. No F&O values were fabricated. Warmup nulls are imputed using training medians only.

## Models and preprocessing

LogisticRegression (max_iter=2000); RandomForest (100 trees, depth 6, minimum leaf 2, n_jobs=1); LightGBM (80 trees, depth 4, 15 leaves, minimum child 10, deterministic CPU, force_col_wise, n_jobs=1). Regression uses Ridge(alpha=1), RandomForestRegressor and LGBMRegressor. sklearn Pipeline persists median imputation and StandardScaler fitted exclusively on each training partition. No transformer, calibration model or parameter search is fitted to validation/test. All-null column selection also uses training only.

LightGBM's [MIT license](https://github.com/lightgbm-org/LightGBM/blob/main/LICENSE) and [CPU settings](https://lightgbm.readthedocs.io/en/latest/Parameters.html) were checked. Runtime container includes libgomp1; no GPU dependency added.

## Chronology and leakage audit

Final holdout uses timestamp groups: 150 train, 60 validation, 85 test before purging. The last five training/validation labels are purged, leaving 145/55/85 (barrier ambiguity can further reduce fitting/metric counts). Walk-forward uses only the 205 development labels ending strictly before holdout start. Initial windows 70/30/30, step 30, three folds each for rolling and expanding. Earlier fold test tails are purged against the next fold test start. All instruments at a timestamp remain on the same side of every boundary.

Audit checks: source feature/label rebuild; prefix invariance; future perturbation; feature allowlist; duplicate instrument timestamps/samples; global time boundaries; disjoint indices; strict label_end < next partition start; train-only imputer/scaler statistics; cross-fold test label separation; no holdout label data in development. Repeated numeric patterns at different timestamps are legitimate observations, not duplicate sample identities. Same instruments recur across time; these tests do not claim transfer to unseen instruments. Overlapping labels within training are permitted; no cross-boundary overlap is permitted.

Saved test evaluation uses exact sample identities, including irregular instrument purges. OOS evaluation rebuilds features and labels from fixture bars. Unknown source controls are rejected rather than discarded. Phase 3 MCP flags remain informational_only=true, training_eligible=false, executable_price=false. Phase 4 rejects MCP provenance and non-synthetic inputs. Provenance and checksums are local engineering controls, not cryptographic proof of the origin of a deliberately relabeled dataset.

| Mode | Fold | Partition | Rows | Start | End | Label end |
| --- | --- | --- | --- | --- | --- | --- |
| holdout | 0 | train | 145 | 2024-01-02T03:46:00+00:00 | 2024-01-02T06:10:00+00:00 | 2024-01-02T06:15:00+00:00 |
| holdout | 0 | validation | 55 | 2024-01-02T06:16:00+00:00 | 2024-01-02T07:10:00+00:00 | 2024-01-02T07:15:00+00:00 |
| holdout | 0 | test | 85 | 2024-01-02T07:16:00+00:00 | 2024-01-02T08:40:00+00:00 | 2024-01-02T08:45:00+00:00 |
| expanding | 0 | train | 65 | 2024-01-02T03:46:00+00:00 | 2024-01-02T04:50:00+00:00 | 2024-01-02T04:55:00+00:00 |
| expanding | 0 | validation | 25 | 2024-01-02T04:56:00+00:00 | 2024-01-02T05:20:00+00:00 | 2024-01-02T05:25:00+00:00 |
| expanding | 0 | test | 25 | 2024-01-02T05:26:00+00:00 | 2024-01-02T05:50:00+00:00 | 2024-01-02T05:55:00+00:00 |
| expanding | 1 | train | 95 | 2024-01-02T03:46:00+00:00 | 2024-01-02T05:20:00+00:00 | 2024-01-02T05:25:00+00:00 |
| expanding | 1 | validation | 25 | 2024-01-02T05:26:00+00:00 | 2024-01-02T05:50:00+00:00 | 2024-01-02T05:55:00+00:00 |
| expanding | 1 | test | 25 | 2024-01-02T05:56:00+00:00 | 2024-01-02T06:20:00+00:00 | 2024-01-02T06:25:00+00:00 |
| expanding | 2 | train | 125 | 2024-01-02T03:46:00+00:00 | 2024-01-02T05:50:00+00:00 | 2024-01-02T05:55:00+00:00 |
| expanding | 2 | validation | 25 | 2024-01-02T05:56:00+00:00 | 2024-01-02T06:20:00+00:00 | 2024-01-02T06:25:00+00:00 |
| expanding | 2 | test | 30 | 2024-01-02T06:26:00+00:00 | 2024-01-02T06:55:00+00:00 | 2024-01-02T07:00:00+00:00 |
| rolling | 0 | train | 65 | 2024-01-02T03:46:00+00:00 | 2024-01-02T04:50:00+00:00 | 2024-01-02T04:55:00+00:00 |
| rolling | 0 | validation | 25 | 2024-01-02T04:56:00+00:00 | 2024-01-02T05:20:00+00:00 | 2024-01-02T05:25:00+00:00 |
| rolling | 0 | test | 25 | 2024-01-02T05:26:00+00:00 | 2024-01-02T05:50:00+00:00 | 2024-01-02T05:55:00+00:00 |
| rolling | 1 | train | 65 | 2024-01-02T04:16:00+00:00 | 2024-01-02T05:20:00+00:00 | 2024-01-02T05:25:00+00:00 |
| rolling | 1 | validation | 25 | 2024-01-02T05:26:00+00:00 | 2024-01-02T05:50:00+00:00 | 2024-01-02T05:55:00+00:00 |
| rolling | 1 | test | 25 | 2024-01-02T05:56:00+00:00 | 2024-01-02T06:20:00+00:00 | 2024-01-02T06:25:00+00:00 |
| rolling | 2 | train | 65 | 2024-01-02T04:46:00+00:00 | 2024-01-02T05:50:00+00:00 | 2024-01-02T05:55:00+00:00 |
| rolling | 2 | validation | 25 | 2024-01-02T05:56:00+00:00 | 2024-01-02T06:20:00+00:00 | 2024-01-02T06:25:00+00:00 |
| rolling | 2 | test | 30 | 2024-01-02T06:26:00+00:00 | 2024-01-02T06:55:00+00:00 | 2024-01-02T07:00:00+00:00 |


## Per-model and per-fold supervised test metrics (SYNTHETIC)

PR-AUC is average precision; undefined ROC/PR for single-class partitions is null. Calibration uses five fixed [0,.2,…,1] bins; the final bin includes 1. ECE is count-weighted absolute prediction/observation discrepancy.

| Target | Mode | Fold | Algorithm | N | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | ECE | CM [[TN,FP],[FN,TP]] | Classes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direction | holdout | 0 | logistic | 85 | 0.96875 | 0.96875 | 0.96875 | 0.997052 | 0.995551 | 0.0194595 | 0.0245484 | [[52, 1], [1, 31]] | {'0': 53, '1': 32} |
| direction | holdout | 0 | random_forest | 85 | 1 | 0.96875 | 0.984127 | 1 | 1 | 0.0129191 | 0.0469454 | [[53, 0], [1, 31]] | {'0': 53, '1': 32} |
| direction | holdout | 0 | lightgbm | 85 | 1 | 0.96875 | 0.984127 | 0.99941 | 0.999053 | 0.011705 | 0.0122572 | [[53, 0], [1, 31]] | {'0': 53, '1': 32} |
| direction | expanding | 0 | logistic | 25 | 1 | 0.846154 | 0.916667 | 1 | 1 | 0.0600432 | 0.125112 | [[12, 0], [2, 11]] | {'0': 12, '1': 13} |
| direction | expanding | 1 | logistic | 25 | 0.625 | 1 | 0.769231 | 1 | 1 | 0.280443 | 0.319565 | [[1, 9], [0, 15]] | {'0': 10, '1': 15} |
| direction | expanding | 2 | logistic | 30 | 1 | 1 | 1 | 1 | 1 | 0.0080153 | 0.0421078 | [[24, 0], [0, 6]] | {'0': 24, '1': 6} |
| direction | expanding | 0 | random_forest | 25 | 1 | 0.846154 | 0.916667 | 1 | 1 | 0.0398694 | 0.0950238 | [[12, 0], [2, 11]] | {'0': 12, '1': 13} |
| direction | expanding | 1 | random_forest | 25 | 0.714286 | 1 | 0.833333 | 1 | 1 | 0.109285 | 0.222008 | [[4, 6], [0, 15]] | {'0': 10, '1': 15} |
| direction | expanding | 2 | random_forest | 30 | 1 | 1 | 1 | 1 | 1 | 0.01376 | 0.0614841 | [[24, 0], [0, 6]] | {'0': 24, '1': 6} |
| direction | expanding | 0 | lightgbm | 25 | 1 | 0.769231 | 0.869565 | 1 | 1 | 0.115908 | 0.117932 | [[12, 0], [3, 10]] | {'0': 12, '1': 13} |
| direction | expanding | 1 | lightgbm | 25 | 0.833333 | 1 | 0.909091 | 1 | 1 | 0.063428 | 0.0996595 | [[7, 3], [0, 15]] | {'0': 10, '1': 15} |
| direction | expanding | 2 | lightgbm | 30 | 1 | 1 | 1 | 1 | 1 | 2.96744e-05 | 0.00162132 | [[24, 0], [0, 6]] | {'0': 24, '1': 6} |
| direction | rolling | 0 | logistic | 25 | 1 | 0.846154 | 0.916667 | 1 | 1 | 0.0600432 | 0.125112 | [[12, 0], [2, 11]] | {'0': 12, '1': 13} |
| direction | rolling | 1 | logistic | 25 | 0.652174 | 1 | 0.789474 | 1 | 1 | 0.2061 | 0.262586 | [[2, 8], [0, 15]] | {'0': 10, '1': 15} |
| direction | rolling | 2 | logistic | 30 | 0.8 | 0.666667 | 0.727273 | 0.9375 | 0.876984 | 0.0666418 | 0.0760696 | [[23, 1], [2, 4]] | {'0': 24, '1': 6} |
| direction | rolling | 0 | random_forest | 25 | 1 | 0.846154 | 0.916667 | 1 | 1 | 0.0398694 | 0.0950238 | [[12, 0], [2, 11]] | {'0': 12, '1': 13} |
| direction | rolling | 1 | random_forest | 25 | 0.75 | 1 | 0.857143 | 0.913333 | 0.943484 | 0.130647 | 0.13902 | [[5, 5], [0, 15]] | {'0': 10, '1': 15} |
| direction | rolling | 2 | random_forest | 30 | 1 | 0.833333 | 0.909091 | 0.986111 | 0.958333 | 0.0574645 | 0.168939 | [[24, 0], [1, 5]] | {'0': 24, '1': 6} |
| direction | rolling | 0 | lightgbm | 25 | 1 | 0.769231 | 0.869565 | 1 | 1 | 0.115908 | 0.117932 | [[12, 0], [3, 10]] | {'0': 12, '1': 13} |
| direction | rolling | 1 | lightgbm | 25 | 0.789474 | 1 | 0.882353 | 0.98 | 0.986627 | 0.101254 | 0.123748 | [[6, 4], [0, 15]] | {'0': 10, '1': 15} |
| direction | rolling | 2 | lightgbm | 30 | 0.208333 | 0.833333 | 0.333333 | 0.652778 | 0.482341 | 0.560621 | 0.600657 | [[5, 19], [1, 5]] | {'0': 24, '1': 6} |
| threshold | holdout | 0 | logistic | 85 | 0.967742 | 1 | 0.983607 | 0.999394 | 0.998925 | 0.0180521 | 0.0237483 | [[54, 1], [0, 30]] | {'0': 55, '1': 30} |
| threshold | holdout | 0 | random_forest | 85 | 0.967742 | 1 | 0.983607 | 0.998788 | 0.997814 | 0.0176884 | 0.0343633 | [[54, 1], [0, 30]] | {'0': 55, '1': 30} |
| threshold | holdout | 0 | lightgbm | 85 | 1 | 1 | 1 | 1 | 1 | 0.00128492 | 0.00432344 | [[55, 0], [0, 30]] | {'0': 55, '1': 30} |
| threshold | expanding | 0 | logistic | 25 | 1 | 0.846154 | 0.916667 | 1 | 1 | 0.0514102 | 0.0936642 | [[12, 0], [2, 11]] | {'0': 12, '1': 13} |
| threshold | expanding | 1 | logistic | 25 | 0.714286 | 1 | 0.833333 | 1 | 1 | 0.188086 | 0.236737 | [[4, 6], [0, 15]] | {'0': 10, '1': 15} |
| threshold | expanding | 2 | logistic | 30 | 1 | 1 | 1 | 1 | 1 | 0.0101178 | 0.0482416 | [[25, 0], [0, 5]] | {'0': 25, '1': 5} |
| threshold | expanding | 0 | random_forest | 25 | 1 | 0.846154 | 0.916667 | 1 | 1 | 0.0576641 | 0.129467 | [[12, 0], [2, 11]] | {'0': 12, '1': 13} |
| threshold | expanding | 1 | random_forest | 25 | 0.714286 | 1 | 0.833333 | 1 | 1 | 0.119385 | 0.231713 | [[4, 6], [0, 15]] | {'0': 10, '1': 15} |
| threshold | expanding | 2 | random_forest | 30 | 1 | 1 | 1 | 1 | 1 | 0.0154351 | 0.0571556 | [[25, 0], [0, 5]] | {'0': 25, '1': 5} |
| threshold | expanding | 0 | lightgbm | 25 | 1 | 0.769231 | 0.869565 | 1 | 1 | 0.119895 | 0.119935 | [[12, 0], [3, 10]] | {'0': 12, '1': 13} |
| threshold | expanding | 1 | lightgbm | 25 | 0.714286 | 1 | 0.833333 | 0.906667 | 0.877451 | 0.224586 | 0.239557 | [[4, 6], [0, 15]] | {'0': 10, '1': 15} |
| threshold | expanding | 2 | lightgbm | 30 | 1 | 1 | 1 | 1 | 1 | 0.00819038 | 0.0174649 | [[25, 0], [0, 5]] | {'0': 25, '1': 5} |
| threshold | rolling | 0 | logistic | 25 | 1 | 0.846154 | 0.916667 | 1 | 1 | 0.0514102 | 0.0936642 | [[12, 0], [2, 11]] | {'0': 12, '1': 13} |
| threshold | rolling | 1 | logistic | 25 | 0.789474 | 1 | 0.882353 | 1 | 1 | 0.155827 | 0.200301 | [[6, 4], [0, 15]] | {'0': 10, '1': 15} |
| threshold | rolling | 2 | logistic | 30 | 0.8 | 0.8 | 0.8 | 0.992 | 0.966667 | 0.0423623 | 0.0878029 | [[24, 1], [1, 4]] | {'0': 25, '1': 5} |
| threshold | rolling | 0 | random_forest | 25 | 1 | 0.846154 | 0.916667 | 1 | 1 | 0.0576641 | 0.129467 | [[12, 0], [2, 11]] | {'0': 12, '1': 13} |
| threshold | rolling | 1 | random_forest | 25 | 0.75 | 1 | 0.857143 | 0.82 | 0.886575 | 0.14832 | 0.102067 | [[5, 5], [0, 15]] | {'0': 10, '1': 15} |
| threshold | rolling | 2 | random_forest | 30 | 1 | 1 | 1 | 1 | 1 | 0.0506201 | 0.201694 | [[25, 0], [0, 5]] | {'0': 25, '1': 5} |
| threshold | rolling | 0 | lightgbm | 25 | 1 | 0.769231 | 0.869565 | 1 | 1 | 0.119895 | 0.119935 | [[12, 0], [3, 10]] | {'0': 12, '1': 13} |
| threshold | rolling | 1 | lightgbm | 25 | 0.652174 | 1 | 0.789474 | 0.666667 | 0.796053 | 0.275057 | 0.273487 | [[2, 8], [0, 15]] | {'0': 10, '1': 15} |
| threshold | rolling | 2 | lightgbm | 30 | 0.208333 | 1 | 0.344828 | 0.776 | 0.535952 | 0.527637 | 0.569651 | [[6, 19], [0, 5]] | {'0': 25, '1': 5} |
| barrier | holdout | 0 | logistic | 85 | 0.939394 | 1 | 0.96875 | 1 | 1 | 0.0122273 | 0.0414977 | [[52, 2], [0, 31]] | {'0': 54, '1': 31} |
| barrier | holdout | 0 | random_forest | 85 | 1 | 1 | 1 | 1 | 1 | 0.0122381 | 0.0470866 | [[54, 0], [0, 31]] | {'0': 54, '1': 31} |
| barrier | holdout | 0 | lightgbm | 85 | 1 | 1 | 1 | 1 | 1 | 0.000111002 | 0.00206669 | [[54, 0], [0, 31]] | {'0': 54, '1': 31} |
| barrier | expanding | 0 | logistic | 25 | 1 | 0.785714 | 0.88 | 1 | 1 | 0.0944898 | 0.146054 | [[11, 0], [3, 11]] | {'0': 11, '1': 14} |
| barrier | expanding | 1 | logistic | 25 | 0.681818 | 1 | 0.810811 | 1 | 1 | 0.203875 | 0.250169 | [[3, 7], [0, 15]] | {'0': 10, '1': 15} |
| barrier | expanding | 2 | logistic | 30 | 1 | 1 | 1 | 1 | 1 | 0.0068136 | 0.0463034 | [[25, 0], [0, 5]] | {'0': 25, '1': 5} |
| barrier | expanding | 0 | random_forest | 25 | 1 | 0.785714 | 0.88 | 1 | 1 | 0.0543746 | 0.128453 | [[11, 0], [3, 11]] | {'0': 11, '1': 14} |
| barrier | expanding | 1 | random_forest | 25 | 0.714286 | 1 | 0.833333 | 1 | 1 | 0.107694 | 0.222606 | [[4, 6], [0, 15]] | {'0': 10, '1': 15} |
| barrier | expanding | 2 | random_forest | 30 | 1 | 1 | 1 | 1 | 1 | 0.0161308 | 0.0603921 | [[25, 0], [0, 5]] | {'0': 25, '1': 5} |
| barrier | expanding | 0 | lightgbm | 25 | 1 | 0.785714 | 0.88 | 1 | 1 | 0.11978 | 0.119913 | [[11, 0], [3, 11]] | {'0': 11, '1': 14} |
| barrier | expanding | 1 | lightgbm | 25 | 0.714286 | 1 | 0.833333 | 0.746667 | 0.760609 | 0.239818 | 0.239885 | [[4, 6], [0, 15]] | {'0': 10, '1': 15} |
| barrier | expanding | 2 | lightgbm | 30 | 1 | 1 | 1 | 1 | 1 | 0.00653104 | 0.0242414 | [[25, 0], [0, 5]] | {'0': 25, '1': 5} |
| barrier | rolling | 0 | logistic | 25 | 1 | 0.785714 | 0.88 | 1 | 1 | 0.0944898 | 0.146054 | [[11, 0], [3, 11]] | {'0': 11, '1': 14} |
| barrier | rolling | 1 | logistic | 25 | 0.789474 | 1 | 0.882353 | 1 | 1 | 0.155827 | 0.200301 | [[6, 4], [0, 15]] | {'0': 10, '1': 15} |
| barrier | rolling | 2 | logistic | 30 | 0.625 | 1 | 0.769231 | 1 | 1 | 0.0712663 | 0.172279 | [[22, 3], [0, 5]] | {'0': 25, '1': 5} |
| barrier | rolling | 0 | random_forest | 25 | 1 | 0.785714 | 0.88 | 1 | 1 | 0.0543746 | 0.128453 | [[11, 0], [3, 11]] | {'0': 11, '1': 14} |
| barrier | rolling | 1 | random_forest | 25 | 0.75 | 1 | 0.857143 | 0.82 | 0.886575 | 0.14832 | 0.102067 | [[5, 5], [0, 15]] | {'0': 10, '1': 15} |
| barrier | rolling | 2 | random_forest | 30 | 0.833333 | 1 | 0.909091 | 1 | 1 | 0.0707616 | 0.245033 | [[24, 1], [0, 5]] | {'0': 25, '1': 5} |
| barrier | rolling | 0 | lightgbm | 25 | 1 | 0.785714 | 0.88 | 1 | 1 | 0.11978 | 0.119913 | [[11, 0], [3, 11]] | {'0': 11, '1': 14} |
| barrier | rolling | 1 | lightgbm | 25 | 0.652174 | 1 | 0.789474 | 0.666667 | 0.796053 | 0.275057 | 0.273487 | [[2, 8], [0, 15]] | {'0': 10, '1': 15} |
| barrier | rolling | 2 | lightgbm | 30 | 0.625 | 1 | 0.769231 | 0.904 | 0.619286 | 0.115108 | 0.177343 | [[22, 3], [0, 5]] | {'0': 25, '1': 5} |

| Target | Mode | Fold | Algorithm | N | MAE | RMSE | R² | Directional accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| future_return | holdout | 0 | ridge | 85 | 0.000682716 | 0.00081489 | 0.995982 | 0.976471 |
| future_return | holdout | 0 | random_forest | 85 | 0.00120053 | 0.00169763 | 0.982564 | 0.988235 |
| future_return | holdout | 0 | lightgbm | 85 | 0.000769377 | 0.00114865 | 0.992018 | 1 |
| future_return | expanding | 0 | ridge | 25 | 0.00214246 | 0.00263445 | 0.964388 | 1 |
| future_return | expanding | 1 | ridge | 25 | 0.00104754 | 0.00128829 | 0.987784 | 0.96 |
| future_return | expanding | 2 | ridge | 30 | 0.000757823 | 0.000934226 | 0.992459 | 1 |
| future_return | expanding | 0 | random_forest | 25 | 0.00352077 | 0.00436422 | 0.90227 | 0.96 |
| future_return | expanding | 1 | random_forest | 25 | 0.00618046 | 0.00722494 | 0.615776 | 0.76 |
| future_return | expanding | 2 | random_forest | 30 | 0.00185254 | 0.00271262 | 0.936423 | 1 |
| future_return | expanding | 0 | lightgbm | 25 | 0.00237838 | 0.00308345 | 0.951215 | 0.96 |
| future_return | expanding | 1 | lightgbm | 25 | 0.00762618 | 0.0099463 | 0.27182 | 0.76 |
| future_return | expanding | 2 | lightgbm | 30 | 0.0014855 | 0.00201095 | 0.96506 | 0.966667 |
| future_return | rolling | 0 | ridge | 25 | 0.00214246 | 0.00263445 | 0.964388 | 1 |
| future_return | rolling | 1 | ridge | 25 | 0.000860312 | 0.00103744 | 0.992078 | 1 |
| future_return | rolling | 2 | ridge | 30 | 0.00177691 | 0.00199905 | 0.965472 | 0.966667 |
| future_return | rolling | 0 | random_forest | 25 | 0.00352077 | 0.00436422 | 0.90227 | 0.96 |
| future_return | rolling | 1 | random_forest | 25 | 0.0112718 | 0.0146425 | -0.578139 | 0.52 |
| future_return | rolling | 2 | random_forest | 30 | 0.00599669 | 0.0079629 | 0.452147 | 0.866667 |
| future_return | rolling | 0 | lightgbm | 25 | 0.00237838 | 0.00308345 | 0.951215 | 0.96 |
| future_return | rolling | 1 | lightgbm | 25 | 0.00860871 | 0.0109016 | 0.125223 | 0.8 |
| future_return | rolling | 2 | lightgbm | 30 | 0.00586254 | 0.00721366 | 0.550393 | 0.866667 |


## Walk-forward aggregates and calibration

| Target | Mode | Algorithm | Metric | Mean | Std | Min | Max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| direction | expanding | lightgbm | brier | 0.0597887 | 0.0473772 | 2.96744e-05 | 0.115908 |
| direction | expanding | lightgbm | expected_calibration_error | 0.073071 | 0.0510703 | 0.00162132 | 0.117932 |
| direction | expanding | lightgbm | f1 | 0.926219 | 0.0546097 | 0.869565 | 1 |
| direction | expanding | logistic | brier | 0.116167 | 0.118087 | 0.0080153 | 0.280443 |
| direction | expanding | logistic | expected_calibration_error | 0.162262 | 0.116277 | 0.0421078 | 0.319565 |
| direction | expanding | logistic | f1 | 0.895299 | 0.095415 | 0.769231 | 1 |
| direction | expanding | random_forest | brier | 0.0543047 | 0.0403115 | 0.01376 | 0.109285 |
| direction | expanding | random_forest | expected_calibration_error | 0.126172 | 0.0691359 | 0.0614841 | 0.222008 |
| direction | expanding | random_forest | f1 | 0.916667 | 0.0680414 | 0.833333 | 1 |
| direction | rolling | lightgbm | brier | 0.259261 | 0.213178 | 0.101254 | 0.560621 |
| direction | rolling | lightgbm | expected_calibration_error | 0.280779 | 0.2262 | 0.117932 | 0.600657 |
| direction | rolling | lightgbm | f1 | 0.695084 | 0.255849 | 0.333333 | 0.882353 |
| direction | rolling | logistic | brier | 0.110928 | 0.0673502 | 0.0600432 | 0.2061 |
| direction | rolling | logistic | expected_calibration_error | 0.154589 | 0.0789462 | 0.0760696 | 0.262586 |
| direction | rolling | logistic | f1 | 0.811138 | 0.0788226 | 0.727273 | 0.916667 |
| direction | rolling | random_forest | brier | 0.0759938 | 0.0393079 | 0.0398694 | 0.130647 |
| direction | rolling | random_forest | expected_calibration_error | 0.134328 | 0.0303576 | 0.0950238 | 0.168939 |
| direction | rolling | random_forest | f1 | 0.8943 | 0.0264556 | 0.857143 | 0.916667 |
| threshold | expanding | lightgbm | brier | 0.117557 | 0.0883586 | 0.00819038 | 0.224586 |
| threshold | expanding | lightgbm | expected_calibration_error | 0.125652 | 0.0907586 | 0.0174649 | 0.239557 |
| threshold | expanding | lightgbm | f1 | 0.900966 | 0.0715726 | 0.833333 | 1 |
| threshold | expanding | logistic | brier | 0.0832045 | 0.0760539 | 0.0101178 | 0.188086 |
| threshold | expanding | logistic | expected_calibration_error | 0.126214 | 0.0803215 | 0.0482416 | 0.236737 |
| threshold | expanding | logistic | f1 | 0.916667 | 0.0680414 | 0.833333 | 1 |
| threshold | expanding | random_forest | brier | 0.0641614 | 0.0426853 | 0.0154351 | 0.119385 |
| threshold | expanding | random_forest | expected_calibration_error | 0.139445 | 0.0716114 | 0.0571556 | 0.231713 |
| threshold | expanding | random_forest | f1 | 0.916667 | 0.0680414 | 0.833333 | 1 |
| threshold | rolling | lightgbm | brier | 0.30753 | 0.168036 | 0.119895 | 0.527637 |
| threshold | rolling | lightgbm | expected_calibration_error | 0.321024 | 0.186647 | 0.119935 | 0.569651 |
| threshold | rolling | lightgbm | f1 | 0.667955 | 0.230814 | 0.344828 | 0.869565 |
| threshold | rolling | logistic | brier | 0.0831999 | 0.0514879 | 0.0423623 | 0.155827 |
| threshold | rolling | logistic | expected_calibration_error | 0.127256 | 0.0517058 | 0.0878029 | 0.200301 |
| threshold | rolling | logistic | f1 | 0.86634 | 0.0489564 | 0.8 | 0.916667 |
| threshold | rolling | random_forest | brier | 0.0855349 | 0.0444892 | 0.0506201 | 0.14832 |
| threshold | rolling | random_forest | expected_calibration_error | 0.144409 | 0.0420229 | 0.102067 | 0.201694 |
| threshold | rolling | random_forest | f1 | 0.924603 | 0.0585906 | 0.857143 | 1 |
| barrier | expanding | lightgbm | brier | 0.122043 | 0.0952526 | 0.00653104 | 0.239818 |
| barrier | expanding | lightgbm | expected_calibration_error | 0.128013 | 0.0882222 | 0.0242414 | 0.239885 |
| barrier | expanding | lightgbm | f1 | 0.904444 | 0.0702025 | 0.833333 | 1 |
| barrier | expanding | logistic | brier | 0.101726 | 0.0806126 | 0.0068136 | 0.203875 |
| barrier | expanding | logistic | expected_calibration_error | 0.147509 | 0.0832339 | 0.0463034 | 0.250169 |
| barrier | expanding | logistic | f1 | 0.896937 | 0.0781592 | 0.810811 | 1 |
| barrier | expanding | random_forest | brier | 0.0593997 | 0.0375489 | 0.0161308 | 0.107694 |
| barrier | expanding | random_forest | expected_calibration_error | 0.13715 | 0.0665084 | 0.0603921 | 0.222606 |
| barrier | expanding | random_forest | f1 | 0.904444 | 0.0702025 | 0.833333 | 1 |
| barrier | rolling | lightgbm | brier | 0.169981 | 0.0743243 | 0.115108 | 0.275057 |
| barrier | rolling | lightgbm | expected_calibration_error | 0.190248 | 0.0633567 | 0.119913 | 0.273487 |
| barrier | rolling | lightgbm | f1 | 0.812901 | 0.0481602 | 0.769231 | 0.88 |
| barrier | rolling | logistic | brier | 0.107194 | 0.0356715 | 0.0712663 | 0.155827 |
| barrier | rolling | logistic | expected_calibration_error | 0.172878 | 0.02215 | 0.146054 | 0.200301 |
| barrier | rolling | logistic | f1 | 0.843861 | 0.0527805 | 0.769231 | 0.882353 |
| barrier | rolling | random_forest | brier | 0.0911522 | 0.0409739 | 0.0543746 | 0.14832 |
| barrier | rolling | random_forest | expected_calibration_error | 0.158518 | 0.0621169 | 0.102067 | 0.245033 |
| barrier | rolling | random_forest | f1 | 0.882078 | 0.0212585 | 0.857143 | 0.909091 |
| future_return | expanding | lightgbm | rmse | 0.00501357 | 0.00351534 | 0.00201095 | 0.0099463 |
| future_return | expanding | random_forest | rmse | 0.00476726 | 0.00186406 | 0.00271262 | 0.00722494 |
| future_return | expanding | ridge | rmse | 0.00161899 | 0.000732443 | 0.000934226 | 0.00263445 |
| future_return | rolling | lightgbm | rmse | 0.00706624 | 0.00319346 | 0.00308345 | 0.0109016 |
| future_return | rolling | random_forest | rmse | 0.00898987 | 0.00425846 | 0.00436422 | 0.0146425 |
| future_return | rolling | ridge | rmse | 0.00189031 | 0.000656497 | 0.00103744 | 0.00263445 |


Holdout probability distributions and calibration bins:

| Target | Algorithm | Mean | Std | Min | Max | Bin counts | Calibration (count, predicted, observed) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| direction | logistic | 0.380679 | 0.448272 | 2.65968e-05 | 0.999994 | [49, 4, 2, 1, 29] | [(49, 0.01987, 0.0), (4, 0.29438, 0.25), (2, 0.53539, 0.5), (1, 0.78407, 1.0), (29, 0.97764, 1.0)] |
| direction | random_forest | 0.376783 | 0.448098 | 0 | 1 | [50, 3, 2, 2, 28] | [(50, 0.01996, 0.0), (3, 0.33688, 0.0), (2, 0.51382, 1.0), (2, 0.74908, 1.0), (28, 0.98187, 1.0)] |
| direction | lightgbm | 0.364213 | 0.480197 | 0.000119059 | 0.999828 | [54, 0, 0, 0, 31] | [(54, 0.0004, 0.01852), (31, 0.99796, 1.0)] |
| threshold | logistic | 0.376689 | 0.454096 | 5.86056e-05 | 0.99999 | [50, 2, 3, 0, 30] | [(50, 0.01368, 0.0), (2, 0.31699, 0.0), (3, 0.47041, 0.33333), (30, 0.97631, 0.96667)] |
| threshold | random_forest | 0.373292 | 0.447578 | 0 | 1 | [50, 3, 2, 2, 28] | [(50, 0.01933, 0.0), (3, 0.2861, 0.0), (2, 0.47899, 0.5), (2, 0.75025, 0.5), (28, 0.98023, 1.0)] |
| threshold | lightgbm | 0.35692 | 0.475909 | 0.000112682 | 0.999838 | [54, 1, 0, 0, 30] | [(54, 0.00042, 0.0), (1, 0.33033, 0.0), (30, 0.99951, 1.0)] |
| barrier | logistic | 0.387334 | 0.45611 | 9.39849e-05 | 0.999989 | [50, 1, 3, 1, 30] | [(50, 0.01828, 0.0), (1, 0.31427, 0.0), (3, 0.49895, 0.0), (1, 0.78181, 1.0), (30, 0.98054, 1.0)] |
| barrier | random_forest | 0.383533 | 0.448984 | 0 | 1 | [49, 2, 3, 1, 30] | [(49, 0.01868, 0.0), (2, 0.25274, 0.0), (3, 0.46016, 0.0), (1, 0.76267, 1.0), (30, 0.96788, 1.0)] |
| barrier | lightgbm | 0.366234 | 0.47974 | 0.000132789 | 0.999827 | [54, 0, 0, 0, 31] | [(54, 0.00283, 0.0), (31, 0.99926, 1.0)] |


## Risk-gated backtests and baseline comparisons (SYNTHETIC)

Historical bars → causal features → predictions → proposed signals → existing RiskEngine → next-bar simulated fill → portfolio. Classifiers use p>=0.5; regressors use predicted return>0; one share, long-only, 5% stop intent. Cash emits nothing; naive follows the previous one-bar close direction; SMA uses existing 5/20 crossover. All methods share calendar, warmup, test window, capital, risk, cost and slippage settings. Warmup bars never trade. No automatic barrier/stop execution; proposed direction may differ from positions after rejection. Final holdings are marked, not force-liquidated.

Initial capital INR 100000; original risk defaults. Illustrative existing equity-delivery cost fixture: brokerage 0.0001/side (cap INR 7), STT sell 0.0003, exchange 0.00002/side, SEBI 0.000001/side, stamp buy 0.00001, GST 10% on brokerage+exchange+SEBI. These are invented test rates. Slippage 2 bps/side plus adverse 0.01 tick rounding; volatility multiplier=0. Turnover is notional traded / initial capital; exposure is average INR notional including initial zero. Winners/losers are net closed sell-fill PnLs. Transaction fees and slippage are separately recorded.

| Method | direction-logistic | cash | naive | sma | direction-random_forest | direction-lightgbm | threshold-logistic | threshold-random_forest | threshold-lightgbm | barrier-logistic | barrier-random_forest | barrier-lightgbm | future_return-ridge | future_return-random_forest | future_return-lightgbm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| total_return | 6.12812e-05 | 0 | 6.67196e-05 | 2.442e-05 | 6.31815e-05 | 6.31815e-05 | 6.31815e-05 | 6.31815e-05 | 6.20764e-05 | 6.57804e-05 | 6.31815e-05 | 6.31815e-05 | 6.44809e-05 | 6.05811e-05 | 6.24814e-05 |
| annualized_return | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined |
| annualized_volatility | 0.000435535 | 0 | 0.00041867 | 0.000453764 | 0.000426483 | 0.000426483 | 0.000426483 | 0.000426483 | 0.000423358 | 0.000425742 | 0.000426483 | 0.000426483 | 0.000426139 | 0.000437634 | 0.000428668 |
| sharpe | 156.425 | undefined | 177.166 | 59.8308 | 164.698 | 164.698 | 164.698 | 164.698 | 163.012 | 171.771 | 164.698 | 164.698 | 168.221 | 153.896 | 162.043 |
| sortino | 720.123 | undefined | 1749.52 | 96.34 | 1066.99 | 1066.99 | 1066.99 | 1066.99 | 1132.59 | 1110.64 | 1066.99 | 1066.99 | 1088.81 | 685.741 | 979.465 |
| max_drawdown | 4.18592e-06 | 0 | 1.44769e-06 | 1.98378e-05 | 2.28565e-06 | 2.28565e-06 | 2.28565e-06 | 2.28565e-06 | 1.58519e-06 | 2.28621e-06 | 2.28565e-06 | 2.28565e-06 | 2.28565e-06 | 4.18592e-06 | 2.28565e-06 |
| calmar | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined |
| win_rate | 1 | undefined | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| loss_rate | 0 | undefined | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| profit_factor | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined |
| average_win | 3.08602 | undefined | 3.33598 | 2.442 | 3.18104 | 3.18104 | 3.18104 | 3.18104 | 6.25156 | 3.31098 | 3.18104 | 3.18104 | 3.24601 | 3.08602 | 3.18104 |
| average_loss | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined |
| expectancy | 3.08602 | undefined | 3.33598 | 2.442 | 3.18104 | 3.18104 | 3.18104 | 3.18104 | 6.25156 | 3.31098 | 3.18104 | 3.18104 | 3.24601 | 3.08602 | 3.18104 |
| turnover | 0.0050355 | 0 | 0.0040631 | 0.0020006 | 0.0050336 | 0.0050336 | 0.0050336 | 0.0050336 | 0.0029697 | 0.0050362 | 0.0050336 | 0.0050336 | 0.0050349 | 0.0050362 | 0.0050343 |
| closed_trade_count | 2 | 0 | 2 | 1 | 2 | 2 | 2 | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 2 |
| average_exposure | 35.9347 | 0 | 35.2577 | 30.77 | 34.8107 | 34.8107 | 34.8107 | 34.8107 | 33.6094 | 37.2129 | 34.8107 | 34.8107 | 36.0101 | 37.0658 | 35.9419 |
| costs | 0.131876 | 0 | 0.118042 | 0.0579998 | 0.131849 | 0.131849 | 0.131849 | 0.131849 | 0.0723635 | 0.131962 | 0.131849 | 0.131849 | 0.131905 | 0.131886 | 0.131859 |
| slippage | 0.13 | 0 | 0.11 | 0.05 | 0.13 | 0.13 | 0.13 | 0.13 | 0.07 | 0.13 | 0.13 | 0.13 | 0.13 | 0.13 | 0.13 |
| fill_count | 5 | 0 | 4 | 2 | 5 | 5 | 5 | 5 | 3 | 5 | 5 | 5 | 5 | 5 | 5 |


Per-fold backtest results (complete risk audits, fills, equity, costs and all remaining ratios are in each report.json):

| Mode | Fold | Method | Return | Drawdown | Fills | Turnover | Fees | Slippage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| expanding | 0 | direction-logistic | 2.19123e-05 | 7.44127e-07 | 2 | 0.0020285 | 0.0587673 | 0.06 |
| expanding | 0 | cash | 0 | 0 | 0 | 0 | 0 | 0 |
| expanding | 0 | naive | 2.74099e-05 | 1.64648e-06 | 2 | 0.002034 | 0.0590055 | 0.06 |
| expanding | 0 | sma | 0 | 0 | 0 | 0 | 0 | 0 |
| expanding | 1 | direction-logistic | 3.58597e-05 | 8.04025e-06 | 1 | 0.0009801 | 0.0140252 | 0.02 |
| expanding | 1 | cash | 0 | 0 | 0 | 0 | 0 | 0 |
| expanding | 1 | naive | 4.31608e-05 | 0 | 1 | 0.0009728 | 0.0139208 | 0.02 |
| expanding | 1 | sma | 3.00589e-05 | 0 | 1 | 0.0009859 | 0.0141082 | 0.02 |
| expanding | 2 | direction-logistic | 9.61222e-07 | 2.13878e-06 | 1 | 0.0009698 | 0.0138778 | 0.02 |
| expanding | 2 | cash | 0 | 0 | 0 | 0 | 0 | 0 |
| expanding | 2 | naive | 8.69478e-07 | 1.59196e-06 | 3 | 0.0030235 | 0.0730522 | 0.08 |
| expanding | 2 | sma | 0 | 0 | 0 | 0 | 0 | 0 |
| expanding | 0 | direction-random_forest | 2.19123e-05 | 7.44127e-07 | 2 | 0.0020285 | 0.0587673 | 0.06 |
| expanding | 1 | direction-random_forest | 3.84601e-05 | 5.43988e-06 | 1 | 0.0009775 | 0.013988 | 0.02 |
| expanding | 2 | direction-random_forest | 9.61222e-07 | 2.13878e-06 | 1 | 0.0009698 | 0.0138778 | 0.02 |
| expanding | 0 | direction-lightgbm | 1.96133e-05 | 7.43133e-07 | 2 | 0.0020262 | 0.0586677 | 0.06 |
| expanding | 1 | direction-lightgbm | 3.97981e-05 | 4.10186e-06 | 3 | 0.0029272 | 0.0701864 | 0.06 |
| expanding | 2 | direction-lightgbm | 9.61222e-07 | 2.13878e-06 | 1 | 0.0009698 | 0.0138778 | 0.02 |
| rolling | 0 | direction-logistic | 2.19123e-05 | 7.44127e-07 | 2 | 0.0020285 | 0.0587673 | 0.06 |
| rolling | 0 | cash | 0 | 0 | 0 | 0 | 0 | 0 |
| rolling | 0 | naive | 2.74099e-05 | 1.64648e-06 | 2 | 0.002034 | 0.0590055 | 0.06 |
| rolling | 0 | sma | 0 | 0 | 0 | 0 | 0 | 0 |
| rolling | 1 | direction-logistic | 3.70599e-05 | 6.84008e-06 | 1 | 0.0009789 | 0.0140081 | 0.02 |
| rolling | 1 | cash | 0 | 0 | 0 | 0 | 0 | 0 |
| rolling | 1 | naive | 4.31608e-05 | 0 | 1 | 0.0009728 | 0.0139208 | 0.02 |
| rolling | 1 | sma | 3.00589e-05 | 0 | 1 | 0.0009859 | 0.0141082 | 0.02 |
| rolling | 2 | direction-logistic | 1.16952e-06 | 1.63048e-06 | 3 | 0.0030232 | 0.0730479 | 0.08 |
| rolling | 2 | cash | 0 | 0 | 0 | 0 | 0 | 0 |
| rolling | 2 | naive | 8.69478e-07 | 1.59196e-06 | 3 | 0.0030235 | 0.0730522 | 0.08 |
| rolling | 2 | sma | 0 | 0 | 0 | 0 | 0 | 0 |
| rolling | 0 | direction-random_forest | 2.19123e-05 | 7.44127e-07 | 2 | 0.0020285 | 0.0587673 | 0.06 |
| rolling | 1 | direction-random_forest | 3.90602e-05 | 4.83979e-06 | 1 | 0.0009769 | 0.0139794 | 0.02 |
| rolling | 2 | direction-random_forest | 2.36142e-06 | 7.38578e-07 | 1 | 0.0009684 | 0.0138578 | 0.02 |
| rolling | 0 | direction-lightgbm | 1.96133e-05 | 7.43133e-07 | 2 | 0.0020262 | 0.0586677 | 0.06 |
| rolling | 1 | direction-lightgbm | 3.97603e-05 | 4.13969e-06 | 1 | 0.0009762 | 0.0139694 | 0.02 |
| rolling | 2 | direction-lightgbm | -3.76139e-05 | 4.07139e-05 | 3 | 0.0029852 | 0.0713905 | 0.07 |
| expanding | 0 | threshold-logistic | 2.19123e-05 | 7.44127e-07 | 2 | 0.0020285 | 0.0587673 | 0.06 |
| expanding | 1 | threshold-logistic | 3.84601e-05 | 5.43988e-06 | 1 | 0.0009775 | 0.013988 | 0.02 |
| expanding | 2 | threshold-logistic | 2.36142e-06 | 7.38578e-07 | 1 | 0.0009684 | 0.0138578 | 0.02 |
| expanding | 0 | threshold-random_forest | 2.19123e-05 | 7.44127e-07 | 2 | 0.0020285 | 0.0587673 | 0.06 |
| expanding | 1 | threshold-random_forest | 3.84601e-05 | 5.43988e-06 | 1 | 0.0009775 | 0.013988 | 0.02 |
| expanding | 2 | threshold-random_forest | 2.36142e-06 | 7.38578e-07 | 1 | 0.0009684 | 0.0138578 | 0.02 |
| expanding | 0 | threshold-lightgbm | 1.96133e-05 | 7.43133e-07 | 2 | 0.0020262 | 0.0586677 | 0.06 |
| expanding | 1 | threshold-lightgbm | 3.84601e-05 | 5.43988e-06 | 1 | 0.0009775 | 0.013988 | 0.02 |
| expanding | 2 | threshold-lightgbm | 2.36142e-06 | 7.38578e-07 | 1 | 0.0009684 | 0.0138578 | 0.02 |
| rolling | 0 | threshold-logistic | 2.19123e-05 | 7.44127e-07 | 2 | 0.0020285 | 0.0587673 | 0.06 |
| rolling | 1 | threshold-logistic | 3.97603e-05 | 4.13969e-06 | 1 | 0.0009762 | 0.0139694 | 0.02 |
| rolling | 2 | threshold-logistic | 1.16952e-06 | 1.63048e-06 | 3 | 0.0030232 | 0.0730479 | 0.08 |
| rolling | 0 | threshold-random_forest | 2.19123e-05 | 7.44127e-07 | 2 | 0.0020285 | 0.0587673 | 0.06 |
| rolling | 1 | threshold-random_forest | 3.90602e-05 | 4.83979e-06 | 1 | 0.0009769 | 0.0139794 | 0.02 |
| rolling | 2 | threshold-random_forest | 2.36142e-06 | 7.38578e-07 | 1 | 0.0009684 | 0.0138578 | 0.02 |
| rolling | 0 | threshold-lightgbm | 1.96133e-05 | 7.43133e-07 | 2 | 0.0020262 | 0.0586677 | 0.06 |
| rolling | 1 | threshold-lightgbm | 3.70599e-05 | 6.84008e-06 | 1 | 0.0009789 | 0.0140081 | 0.02 |
| rolling | 2 | threshold-lightgbm | -3.76139e-05 | 4.07139e-05 | 3 | 0.0029852 | 0.0713905 | 0.07 |
| expanding | 0 | barrier-logistic | 2.19123e-05 | 7.44127e-07 | 2 | 0.0020285 | 0.0587673 | 0.06 |
| expanding | 1 | barrier-logistic | 3.786e-05 | 6.03997e-06 | 1 | 0.0009781 | 0.0139966 | 0.02 |
| expanding | 2 | barrier-logistic | 2.36142e-06 | 7.38578e-07 | 1 | 0.0009684 | 0.0138578 | 0.02 |
| expanding | 0 | barrier-random_forest | 2.19123e-05 | 7.44127e-07 | 2 | 0.0020285 | 0.0587673 | 0.06 |
| expanding | 1 | barrier-random_forest | 3.84601e-05 | 5.43988e-06 | 1 | 0.0009775 | 0.013988 | 0.02 |
| expanding | 2 | barrier-random_forest | 2.36142e-06 | 7.38578e-07 | 1 | 0.0009684 | 0.0138578 | 0.02 |
| expanding | 0 | barrier-lightgbm | 2.19123e-05 | 7.44127e-07 | 2 | 0.0020285 | 0.0587673 | 0.06 |
| expanding | 1 | barrier-lightgbm | 3.84601e-05 | 5.43988e-06 | 1 | 0.0009775 | 0.013988 | 0.02 |
| expanding | 2 | barrier-lightgbm | 2.36142e-06 | 7.38578e-07 | 1 | 0.0009684 | 0.0138578 | 0.02 |
| rolling | 0 | barrier-logistic | 2.19123e-05 | 7.44127e-07 | 2 | 0.0020285 | 0.0587673 | 0.06 |
| rolling | 1 | barrier-logistic | 3.97603e-05 | 4.13969e-06 | 1 | 0.0009762 | 0.0139694 | 0.02 |
| rolling | 2 | barrier-logistic | -3.11161e-06 | 6.21161e-06 | 7 | 0.0071245 | 0.191161 | 0.2 |
| rolling | 0 | barrier-random_forest | 2.19123e-05 | 7.44127e-07 | 2 | 0.0020285 | 0.0587673 | 0.06 |
| rolling | 1 | barrier-random_forest | 3.90602e-05 | 4.83979e-06 | 1 | 0.0009769 | 0.0139794 | 0.02 |
| rolling | 2 | barrier-random_forest | 7.69464e-07 | 2.33054e-06 | 3 | 0.0030236 | 0.0730536 | 0.08 |
| rolling | 0 | barrier-lightgbm | 2.19123e-05 | 7.44127e-07 | 2 | 0.0020285 | 0.0587673 | 0.06 |
| rolling | 1 | barrier-lightgbm | 3.70599e-05 | 6.84008e-06 | 1 | 0.0009789 | 0.0140081 | 0.02 |
| rolling | 2 | barrier-lightgbm | -6.2993e-07 | 3.72993e-06 | 3 | 0.0030222 | 0.072993 | 0.08 |
| expanding | 0 | future_return-ridge | 2.59106e-05 | 7.45857e-07 | 2 | 0.0020325 | 0.0589406 | 0.06 |
| expanding | 1 | future_return-ridge | 4.20606e-05 | 1.83937e-06 | 1 | 0.0009739 | 0.0139365 | 0.02 |
| expanding | 2 | future_return-ridge | 9.61222e-07 | 2.13878e-06 | 1 | 0.0009698 | 0.0138778 | 0.02 |
| expanding | 0 | future_return-random_forest | 2.40114e-05 | 7.45035e-07 | 2 | 0.0020306 | 0.0588583 | 0.06 |
| expanding | 1 | future_return-random_forest | 3.84601e-05 | 5.43988e-06 | 1 | 0.0009775 | 0.013988 | 0.02 |
| expanding | 2 | future_return-random_forest | 9.61222e-07 | 2.13878e-06 | 1 | 0.0009698 | 0.0138778 | 0.02 |
| expanding | 0 | future_return-lightgbm | 2.40114e-05 | 7.45035e-07 | 2 | 0.0020306 | 0.0588583 | 0.06 |
| expanding | 1 | future_return-lightgbm | 3.84601e-05 | 5.43988e-06 | 1 | 0.0009775 | 0.013988 | 0.02 |
| expanding | 2 | future_return-lightgbm | -6.30737e-07 | 3.73074e-06 | 3 | 0.003025 | 0.0730737 | 0.08 |
| rolling | 0 | future_return-ridge | 2.59106e-05 | 7.45857e-07 | 2 | 0.0020325 | 0.0589406 | 0.06 |
| rolling | 1 | future_return-ridge | 4.28607e-05 | 1.03925e-06 | 1 | 0.0009731 | 0.0139251 | 0.02 |
| rolling | 2 | future_return-ridge | 2.36142e-06 | 7.38578e-07 | 1 | 0.0009684 | 0.0138578 | 0.02 |
| rolling | 0 | future_return-random_forest | 2.40114e-05 | 7.45035e-07 | 2 | 0.0020306 | 0.0588583 | 0.06 |
| rolling | 1 | future_return-random_forest | 1.16945e-05 | 5.43988e-06 | 3 | 0.0029541 | 0.070554 | 0.07 |
| rolling | 2 | future_return-random_forest | -4.43047e-06 | 7.53047e-06 | 3 | 0.003026 | 0.0730474 | 0.08 |
| rolling | 0 | future_return-lightgbm | 2.40114e-05 | 7.45035e-07 | 2 | 0.0020306 | 0.0588583 | 0.06 |
| rolling | 1 | future_return-lightgbm | 3.81975e-05 | 5.7025e-06 | 3 | 0.0029302 | 0.0702497 | 0.06 |
| rolling | 2 | future_return-lightgbm | -6.2993e-07 | 3.72993e-06 | 3 | 0.0030222 | 0.072993 | 0.08 |


Annualized return and Calmar are undefined because every window is shorter than one year (94500 minute periods). Sharpe, Sortino and annualized volatility are mechanical diagnostics on a tiny synthetic sample, without inferential meaning. Fold portfolios reset; overlapping rolling/expanding runs must not be pooled into a portfolio curve.


## Ranking for FUTURE REAL-DATA EVALUATION only

The following priority is based on SYNTHETIC data only. Eligibility requires passed leakage and reproducibility checks. Fixed ranking key: worst across rolling/expanding (mean Brier + fold Brier std), then holdout Brier, holdout ECE, holdout turnover, runtime. This emphasizes calibration and stability; returns and accuracy are not ranking inputs.

| Rank | Classifier | WF mean+std (worst mode) | Holdout Brier | Holdout ECE | Turnover | Fit seconds |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | random_forest | 0.115302 | 0.0129191 | 0.0469454 | 0.0050336 | 0.210546 |
| 2 | logistic | 0.234254 | 0.0194595 | 0.0245484 | 0.0050355 | 0.100393 |
| 3 | lightgbm | 0.472439 | 0.011705 | 0.0122572 | 0.0050336 | 0.0925546 |


Carry all three modest CPU classifiers into licensed-data evaluation, with `random_forest` first in this synthetic engineering ordering. Retain Logistic Regression as the interpretable reference regardless of rank. Retain Ridge as the regression reference and compare tree regressors on licensed data. Single-session scores cannot establish market robustness.


## Registry, artifacts and runtime

Run root: `data/models/PHASE4-SYNTHETIC-63d2625cfabd48a2877b08f98ae00978`. Summary: `summary.json`; source/config checksums: `checksums.json`; imported immutable bars: `imports/`; datasets: `datasets/`; models: `registry/<model_id>/{model.joblib,metadata.json}`; each `registry/runs/SYNTHETIC-*/` has training_config, metrics, walk_forward, predictions, leakage_audit and checksums JSON. Simulations: `backtests/<mode>/<fold>/<method>/report.json` and report.md. All runtime artifacts are ignored; no model binaries or generated datasets are committed. UUIDs identify local artifacts, not stable model contents.

| Target | Mode | Fold | Algorithm | Model ID | Fit seconds |
| --- | --- | --- | --- | --- | --- |
| direction | holdout | 0 | logistic | d3752156-562a-46d9-b816-18b1da894700 | 0.100393 |
| direction | holdout | 0 | random_forest | 07a3e64a-d7e4-41fb-b8f6-63d47905b46a | 0.210546 |
| direction | holdout | 0 | lightgbm | ff0bb514-0918-4da6-9810-af838625f347 | 0.0925546 |
| direction | expanding | 0 | logistic | a427edad-8802-4d92-b2b1-ba3b5270db5e | 0.108042 |
| direction | expanding | 1 | logistic | 4b02d53b-fe20-459a-ae4c-2a9335b63d36 | 0.102258 |
| direction | expanding | 2 | logistic | a10f1c69-895d-4371-b089-a8519fd84e28 | 0.0889196 |
| direction | expanding | 0 | random_forest | b67fd632-fc99-4260-b013-33e2b37e8ed4 | 0.169717 |
| direction | expanding | 1 | random_forest | c43da87c-0b9b-4eeb-a984-379f864740c8 | 0.160278 |
| direction | expanding | 2 | random_forest | db7f8fdc-3a1b-4678-bd5d-ae38999e46be | 0.162844 |
| direction | expanding | 0 | lightgbm | f9ea7516-2bd1-4dd3-b70e-05dbc3e23672 | 0.05896 |
| direction | expanding | 1 | lightgbm | e5ccbfdd-d02d-47cb-81c0-a6548fd40f4e | 0.0634219 |
| direction | expanding | 2 | lightgbm | c9deed75-3f05-47fe-b116-57a00d105283 | 0.0667741 |
| direction | rolling | 0 | logistic | 9f12ad4c-f2f2-47c4-a46f-293f4923b6ec | 0.0742835 |
| direction | rolling | 1 | logistic | cd2ace77-25a1-45e3-96bd-a730e83a1bdd | 0.0774561 |
| direction | rolling | 2 | logistic | eae02655-6d0d-4874-bf83-0273ec68bd05 | 0.0799217 |
| direction | rolling | 0 | random_forest | f6e301aa-598d-44ca-8ee1-5a72cbeee4b0 | 0.17438 |
| direction | rolling | 1 | random_forest | a7e18294-b794-472f-9714-437d1c6e77a5 | 0.164323 |
| direction | rolling | 2 | random_forest | 7f796a48-30c9-4062-862b-66e5773998cf | 0.161684 |
| direction | rolling | 0 | lightgbm | 0df47bc6-eee3-4cc4-a4bd-e2888eb52386 | 0.0597052 |
| direction | rolling | 1 | lightgbm | 4c8db944-a3b1-49c1-b3eb-3bc8019982b4 | 0.0606859 |
| direction | rolling | 2 | lightgbm | 7a5b069f-6d81-4890-95af-df559d923575 | 0.0640607 |
| threshold | holdout | 0 | logistic | d3b3d288-cb8d-431e-a790-e6c61b5d589c | 0.0893608 |
| threshold | holdout | 0 | random_forest | 4cb513f9-698a-451b-8a58-1b9d357639b3 | 0.180912 |
| threshold | holdout | 0 | lightgbm | 076e429b-f926-478b-bbba-dc4614c56a2b | 0.0780998 |
| threshold | expanding | 0 | logistic | 0a563980-3c13-48ae-86f3-3158fc56927c | 0.0865147 |
| threshold | expanding | 1 | logistic | 2715d9d6-4731-4c0f-8e82-a26c563b90c0 | 0.0793598 |
| threshold | expanding | 2 | logistic | 01415083-f3fe-4741-8cd7-ec3106752469 | 0.0756153 |
| threshold | expanding | 0 | random_forest | 34667ecc-d271-45e4-99aa-9f29defe2314 | 0.176167 |
| threshold | expanding | 1 | random_forest | d112e89b-a626-4df9-8229-72fb043e1904 | 0.164948 |
| threshold | expanding | 2 | random_forest | 2e699489-b144-4c3c-829b-ab0b324b1a15 | 0.166668 |
| threshold | expanding | 0 | lightgbm | 67787d16-3e26-4379-bec0-bfe83fe80129 | 0.0578077 |
| threshold | expanding | 1 | lightgbm | 58fe1d2e-12d7-4e9e-a6f8-dc707f48c3cf | 0.0645645 |
| threshold | expanding | 2 | lightgbm | 85d5a9df-693f-413c-8a85-ff39f9bd3317 | 0.0703796 |
| threshold | rolling | 0 | logistic | e6bf8a3c-88ff-4470-8551-5ee6e2bf4a08 | 0.0730035 |
| threshold | rolling | 1 | logistic | 7eeb325f-7ef9-46ae-b5c7-77cdd36c1f3a | 0.074938 |
| threshold | rolling | 2 | logistic | 6c6959af-b707-43a5-8d3c-e61ec689e6b3 | 0.0782553 |
| threshold | rolling | 0 | random_forest | 85e3c14c-0dc8-4708-9c21-4b4ff3c67171 | 0.172594 |
| threshold | rolling | 1 | random_forest | db4f5521-a3f4-4c7e-9e12-e0b55ccf970e | 0.160016 |
| threshold | rolling | 2 | random_forest | 758f3d30-0a32-492f-bc3f-11d2296f2f22 | 0.158826 |
| threshold | rolling | 0 | lightgbm | b980f8b9-60e5-4b9a-aa9a-6ce96308fb85 | 0.0580459 |
| threshold | rolling | 1 | lightgbm | c0c6d315-4149-432f-b0c0-fc66546d14de | 0.0611273 |
| threshold | rolling | 2 | lightgbm | a9f225e8-8b6b-4152-bab6-7f78e6ad4c53 | 0.058591 |
| barrier | holdout | 0 | logistic | 2d69cbb0-8213-4b1a-93e8-19755d748c1a | 0.0940509 |
| barrier | holdout | 0 | random_forest | 402c7bf0-6e03-4cda-90a8-cd74719948b9 | 0.176269 |
| barrier | holdout | 0 | lightgbm | f82e7258-43b6-41ca-bd19-5352c3b9385a | 0.0777648 |
| barrier | expanding | 0 | logistic | 6eb5a829-db4c-4bce-be56-6f541a3d799c | 0.0725997 |
| barrier | expanding | 1 | logistic | 1d1b1dc5-c028-4d36-ba06-974b45420a93 | 0.0767212 |
| barrier | expanding | 2 | logistic | bb6e2c53-4829-4600-9b46-905a1fbce1ab | 0.077167 |
| barrier | expanding | 0 | random_forest | 16bed0e0-f387-4c3d-978a-027f1214f821 | 0.169961 |
| barrier | expanding | 1 | random_forest | 468f63b3-e23e-4930-8297-bad849a88e97 | 0.16038 |
| barrier | expanding | 2 | random_forest | ee16b695-2cc2-4446-8982-496767e1066f | 0.161035 |
| barrier | expanding | 0 | lightgbm | b999f33f-b3f9-4a46-a6c1-5a163eebca0f | 0.057925 |
| barrier | expanding | 1 | lightgbm | 2c9d64d7-8b36-490e-b379-8f6aa1c2167c | 0.0622328 |
| barrier | expanding | 2 | lightgbm | f869bbfa-ec8a-43f4-aa96-5f28dca381ee | 0.0657594 |
| barrier | rolling | 0 | logistic | 4be87376-3e88-4f15-81e1-969173d2dffe | 0.0778535 |
| barrier | rolling | 1 | logistic | fb059ba7-95f3-45f1-964f-7938d537d171 | 0.0780869 |
| barrier | rolling | 2 | logistic | afbcd06c-1a28-40c2-bb8e-d925c3df20b6 | 0.0816283 |
| barrier | rolling | 0 | random_forest | 5df5aa30-a7c3-4c64-b727-8af99f6e008a | 0.196012 |
| barrier | rolling | 1 | random_forest | 41fb270c-6847-4a47-9cc9-b2838b3cfb2f | 0.159414 |
| barrier | rolling | 2 | random_forest | 5a54bd4a-d39f-41a0-964f-198b524496e9 | 0.160208 |
| barrier | rolling | 0 | lightgbm | ffdfabce-b6fd-4b0c-8788-11cf80efdf81 | 0.056804 |
| barrier | rolling | 1 | lightgbm | c5de0995-9c99-4dfa-be5b-2c4597f7bd2b | 0.062005 |
| barrier | rolling | 2 | lightgbm | f422f811-bb38-4027-a78a-2a4e90594fd0 | 0.0586102 |
| future_return | holdout | 0 | ridge | ea14dce9-66f7-4e29-a469-ea775e02456a | 0.0434726 |
| future_return | holdout | 0 | random_forest | cd889055-14a2-4ba4-aaa8-2f149ddff1f8 | 0.20219 |
| future_return | holdout | 0 | lightgbm | 82f99a8c-a08d-4344-b39e-7c7e1c777de9 | 0.0642229 |
| future_return | expanding | 0 | ridge | 562bb49b-55bd-48e5-8266-625e1c2ca063 | 0.0321671 |
| future_return | expanding | 1 | ridge | 7368e4b8-f5ba-4f57-a970-58db87dd0479 | 0.0322224 |
| future_return | expanding | 2 | ridge | 3a629f48-0f25-4f2d-ab8f-36e2f55b1032 | 0.0345555 |
| future_return | expanding | 0 | random_forest | 6317ef03-12ea-4d8e-b217-3748b6e32e69 | 0.140255 |
| future_return | expanding | 1 | random_forest | 297cb376-cf7d-41ee-abcb-8c45360710a6 | 0.156945 |
| future_return | expanding | 2 | random_forest | 78668963-a028-47d3-8cdb-424eb9617280 | 0.178061 |
| future_return | expanding | 0 | lightgbm | 3c6e31e4-1b7b-4c8d-8d51-0002fc5b6b40 | 0.0412037 |
| future_return | expanding | 1 | lightgbm | 7cdcc9c2-ae4e-4116-b6f5-ffbd3010975f | 0.0464337 |
| future_return | expanding | 2 | lightgbm | 1b247d76-343c-482b-a135-54affa8eac34 | 0.0499812 |
| future_return | rolling | 0 | ridge | fb699f79-af10-4a41-95ca-3df24a0c7ace | 0.0317659 |
| future_return | rolling | 1 | ridge | 0f98a4cc-32bb-4ba6-b6fb-fc5dc2e32dca | 0.0317431 |
| future_return | rolling | 2 | ridge | d041bc9b-46b1-49d7-b404-df3180f1a8a7 | 0.0317312 |
| future_return | rolling | 0 | random_forest | a4616188-6ee9-4c55-a5e7-17d8952277bd | 0.14025 |
| future_return | rolling | 1 | random_forest | c7f9edb8-842a-4467-9365-a4622b74c4ee | 0.143881 |
| future_return | rolling | 2 | random_forest | 3f3b85fc-e40b-42ec-9d35-0cb3beedff34 | 0.142454 |
| future_return | rolling | 0 | lightgbm | 2a066e66-6658-4de8-a5af-dbf89aa4b999 | 0.0405611 |
| future_return | rolling | 1 | lightgbm | 0e626c0a-6749-43d5-acdb-89e212f7156a | 0.0427361 |
| future_return | rolling | 2 | lightgbm | 1a767475-2365-4a8f-ad81-2ee6d083325a | 0.0419326 |


Wall runtime: 174.785 seconds. Summed recorded model fit/evaluation time: 8.338 seconds. Process peak RSS: 223.46 MiB (Linux ru_maxrss, process-wide; not per-model allocation). These timings describe this environment only. Repeated holdout prediction checks: 12 passed.


## Limitations and Phase 5

This fixture has one artificial instrument/session, smooth invented patterns, no corporate actions, liquidity regimes or real spreads; apparent prediction quality is not evidence of a tradeable edge. Calibration diagnostics are small-sample estimates; no recalibration is claimed. Single-class/constant-target undefined metrics stay null. OHLC barrier ties are unknowable. Existing simulation lacks automatic stops, market impact, derivatives margin and exchange matching. Risk rejection remains authoritative.

Once authorized/licensed NSE historical data is supplied, first add explicit license/provenance approval and a reviewed ingestion policy; do not disable the MCP boundary. Verify sessions, timezone/availability timestamps, corporate actions, symbol/contract lifetimes, survivorship, F&O underlying alignment, and realistic exchange/broker costs. Freeze a multi-regime final holdout, repeat purged rolling/expanding evaluation and instrument-held-out tests, add uncertainty estimates and benchmark comparisons. Fit any calibrator only on permitted development observations. Phase 5 should establish those real-data contracts, monitoring/drift and offline replay acceptance criteria while keeping execution paper. No live brokerage integration is recommended by this phase.


## Final verification and review

- `./scripts/check.sh`: passed; **203 passed, 2 optional service tests skipped** in
  the main suite, 91% statement coverage. The two service tests separately passed
  via `./scripts/check-services.sh` using isolated rootless PostgreSQL/Valkey
  containers and local sockets. No live NSE MCP checks were invoked.
- Explicit `uv run ruff format --check src tests scripts`, `uv run ruff check src
  tests scripts`, `uv run mypy src`, and `uv build`: passed. Distribution inspection
  verified all 62 application modules and no local runtime state.
- Rootless Podman build: passed, image `localhost/trading-agent:phase4`, ID
  `010b039af5b88e9971dfbc404ed3afb3092e5bddf31344b584eb391d95edee0d`.
  An isolated `--network none` image smoke check imported LightGBM 4.7.0 and verified
  `execution_mode=paper`.
- Final artifact audit: **84 registry model checksums**, **105 backtests**, and
  **465 artifact checksums** verified. Every model recorded the same clean training
  commit `af5d3beff0efda0721dbfbc7857fca6ed95cd07b`, with `git_dirty=false`.
  Validation/test prediction identities matched each saved partition exactly;
  all compared methods shared the same test windows and 2-bps slippage.
- Twelve repeated holdout fits matched within 1e-12. Across three complete runs,
  all 84 model metric sets and all 105 backtest metric sets matched exactly.
  Model IDs, timestamps, binary checksums and runtime measurements are intentionally
  not treated as deterministic outputs.
- Aggregate fold metrics are unweighted means with population standard deviations.
  Full validation metrics, preprocessing statistics, hyperparameters and sample
  identities are in each model's `metadata.json`; full per-fold metrics and
  probabilities are in the run JSON artifacts.

Independent read-only review identified four Important findings, all fixed with
regression tests that first failed: exact irregular-instrument test membership,
OOS causal reconstruction, accidental non-synthetic substring authorization, and
silently discarded source eligibility controls. A final policy regression also
verified that the standalone backtest CLI rejects MCP provenance before simulation.
All tests use synthetic fixtures; relabeling a fixture for a rejection test does
not involve importing an NSE MCP response. Unknown bar/row fields are rejected.

No Critical findings or deferred Minor findings remain. Market usefulness and
investment suitability are explicitly unclaimed; cryptographic origin attestation
is outside the local checksum controls. The primary agent verified this report and
release evidence after the independent code review. Coverage subprocess state is
ignored, alongside all model/data/backtest runtime artifacts.

Work remains on `phase4-model-training`; previous phase history and tag object IDs
are preserved. No merge, history rewrite, tag change, brokerage connection or live
execution was performed. The only remaining external completion step after the
final documentation commit and clean-tree check is the user-requested ntfy notice.
