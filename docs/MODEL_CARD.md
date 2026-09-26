# Model Card — Phase 4 Synthetic Research

> [!WARNING]
> **SYNTHETIC ENGINEERING VALIDATION ONLY.** These results come from the repository's deterministic synthetic fixture and are **not evidence of expected market performance, profitability, or live-trading suitability**.

## Purpose

Phase 4 validates the model-training and evaluation machinery before licensed historical NSE data is introduced. It tests causal feature generation, purged chronological splits, walk-forward evaluation, train-only preprocessing, model serialization, registry integrity, calibration diagnostics, and risk-gated backtesting.

The full experiment is documented in [PHASE4_MODEL_TRAINING_REPORT.md](../PHASE4_MODEL_TRAINING_REPORT.md).

## Direction-classifier recall

For the **Random Forest direction classifier**, recall measures the share of actual positive labels that the model correctly identifies:

`recall = TP / (TP + FN)`

| Evaluation | N | Precision | Recall | F1 | ROC-AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Holdout | 85 | 100.0% | **96.9%** | 98.4% | 1.000 | 0.0129 |
| Expanding fold 0 | 25 | 100.0% | **84.6%** | 91.7% | 1.000 | 0.0399 |
| Expanding fold 1 | 25 | 71.4% | **100.0%** | 83.3% | 1.000 | 0.1093 |
| Expanding fold 2 | 30 | 100.0% | **100.0%** | 100.0% | 1.000 | 0.0138 |
| Rolling fold 0 | 25 | 100.0% | **84.6%** | 91.7% | 1.000 | 0.0399 |
| Rolling fold 1 | 25 | 75.0% | **100.0%** | 85.7% | 0.913 | 0.1306 |
| Rolling fold 2 | 30 | 100.0% | **83.3%** | 90.9% | 0.986 | 0.0575 |

The three rolling folds average roughly **89.3% recall**. The observed range is **83.3%–100%** across rolling windows.

### What 83.3% recall means

On rolling fold 2 the confusion matrix is:

`[[TN=24, FP=0], [FN=1, TP=5]]`

So the model detected **5 of 6** synthetic positive opportunities, missed **1**, and produced **0 false positives** on that 30-sample fold.

The 85-sample holdout detected **31 of 32** positives, for **96.9% recall**, with no false positives.

## There is no universal "industry standard" recall

Systematic-trading models are not normally accepted or rejected on a single recall threshold. A high-recall classifier can still be unusable if it creates too many false positives, is poorly calibrated, overfits one market regime, or loses its edge after spread, slippage, fees, turnover, and risk constraints.

For this project, model quality is therefore assessed jointly across:

- precision, recall, F1, ROC-AUC and PR-AUC;
- probability calibration and Brier score;
- chronological holdout and purged rolling/expanding walk-forward stability;
- leakage audits and train-only preprocessing;
- turnover, transaction costs and slippage;
- risk-gated backtest behavior, drawdown and expectancy;
- reproducibility across datasets, symbols, dates and market regimes.

This is an **evaluation framework**, not a claim that a particular metric value represents an industry certification threshold.

## Expected real-market behavior

The synthetic fixture is deliberately structured and much cleaner than live financial markets. These Phase 4 scores should therefore be treated as a test of the software pipeline, not as a forecast of real-data accuracy.

When licensed historical NSE data is introduced, the key question is not whether the model preserves a 96.9% synthetic holdout recall. The useful question is whether a statistically meaningful signal remains stable across:

- multiple instruments and sectors;
- bull, bear, sideways and high-volatility regimes;
- different expiry cycles for derivatives;
- longer out-of-sample periods;
- realistic transaction costs and slippage;
- repeated walk-forward windows.

A real-data model should only advance after it shows robust out-of-sample behavior and survives paper-trading validation under the deterministic risk engine.

## Model set

Phase 4 generates **84 local model artifacts** across:

| Task | Algorithms |
| --- | --- |
| Classification | Logistic Regression, Random Forest, LightGBM |
| Regression | Ridge, Random Forest Regressor, LightGBM Regressor |

The experiment covers direction, threshold, barrier and future-return objectives across holdout, rolling and expanding evaluation windows.

## Safety boundary

Model predictions are proposals, not execution authority.

```text
Market / historical data
        ↓
Causal features
        ↓
Model prediction
        ↓
Proposed signal
        ↓
Deterministic risk engine
        ↓
Paper execution
```

NSE MCP remains **informational only**, is not training eligible, and is not an executable price source. HDFC SKY live execution remains disabled.

## Source of truth

Exact per-fold metrics, target definitions, dataset checksums, feature versions, preprocessing metadata, model IDs and backtest outputs are in [PHASE4_MODEL_TRAINING_REPORT.md](../PHASE4_MODEL_TRAINING_REPORT.md).


## Phase 5A reliability qualification

The Phase 4 figures above remain historical synthetic engineering observations.
Phase 5A does not recompute those experiments or claim expected trading returns.
Models now carry training-only distribution/calibration/performance references.
Healthy/watch/quarantined diagnostics describe observed reliability under a
versioned policy, not economic usefulness. Training-reference metrics are optimistic
in-sample diagnostics; they do not replace purged out-of-sample validation.

Feature/schema integrity and current model health gate model-backed paper proposals.
Quarantined models cannot create paper orders, even with high-confidence predictions.
A healthy model with a high-confidence score and valid quote still faces every
existing deterministic risk policy. Research evidence cannot manufacture prices,
account facts or model metrics. NSE MCP remains informational-only.

Newly trained artifacts are challengers. Reviewed explicit promotion is required
to become champion; monitoring never automatically promotes or deploys retraining.
A quarantined artifact remains quarantined; create and validate a new challenger.
See [MODEL_DRIFT.md](MODEL_DRIFT.md), [MODEL_LIFECYCLE.md](MODEL_LIFECYCLE.md) and
[GROUNDING.md](GROUNDING.md) for exact policy, methodology and limitations.
