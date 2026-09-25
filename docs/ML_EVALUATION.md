# Evaluation conventions and limits

Classification reports contain precision, recall, F1 at probability threshold 0.5,
ROC AUC, PR average precision, fixed [0,1] confusion matrix, Brier score and five
probability-bin calibration summaries. ROC/PR AUC are null for one-class evaluation
sets; undefined precision/recall/F1 are explicitly zero. Empty evaluation fails.
Barrier evaluation reports excluded ambiguous/unreached rows; it measures conditional
first-hit discrimination, not unconditional probability of reaching a target.

The consequence diagnostic averages predicted-positive future returns less the
specified round-trip `cost_bps`. Negative predictions contribute zero. It prominently
states NOT A BACKTEST: observations can overlap, and it models neither capital,
portfolio accounting, spread, fills, exchange constraints nor risk gates. Use the
separate risk-gated backtester for economic comparisons, including baseline strategies
and the same costs and execution rules. Synthetic results validate software only.

Hold test data aside from feature design and hyperparameter selection. Purging prevents
label-horizon overlap at boundaries; it does not remove all serial correlation or
selection bias. Historical adjustment quality, survivorship, universe selection,
contract rolls and availability-time errors remain data responsibilities. Always
retain dataset versions and provenance; never replace missing derivative fields with
plausible guessed values. Training tests verify prefix causality, instrument isolation,
barrier ambiguity, purged boundaries, deterministic predictions, training-only
imputation, registry trust, corruption detection and inference schema validation.
