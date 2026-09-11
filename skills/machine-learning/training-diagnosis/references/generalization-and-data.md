# Generalization and Data

Read when train and validation diverge, calibration or per-class metrics
degrade, or an input-distribution shift is suspected.

## Verify the evaluation before the model

Most "overfitting" reports are evaluation defects: evaluation mode not
set (dropout or normalization in training mode), different preprocessing
between train and validation, leakage, a validation set too small for
the reported difference, a changed evaluation set or metric definition.
Check these first; then compare per subgroup (source, class, language,
length) rather than one aggregate.

## Ranked causes for a widening gap

1. Capacity or training length large relative to the data.
2. Insufficient regularization (weight decay, augmentation, dropout).
3. Train/validation distribution mismatch.
4. Validation noise (small set; report a confidence bound).
5. Leakage or preprocessing inconsistency.

Fix in that order after the evaluation checks; early stopping with
restore-best on a smoothed validation metric is the baseline remedy.

## Calibration and per-class collapse

An accuracy that holds while validation loss or calibration worsens
means growing overconfidence. Report calibration with a reliability
diagram and a proper scoring rule (log loss, Brier), not a single
binned error. A minority class whose recall collapsed while overall
accuracy held is caught only by per-class recall and precision-recall
curves; on imbalanced data the ROC curve is optimistic.

## Data shift

Distinguish three questions: did the input distribution change (data
drift), did the input-to-label relation change (concept drift), did
performance degrade? Drift tests (KS, Wasserstein, chi-square,
Jensen–Shannon, PSI, MMD on embeddings) answer only the first; a
detected shift then needs an explanation — which subgroup, which
feature — before any fix. Library default thresholds are tool defaults.

## Hard examples are not bad examples

A heavy loss tail may be valuable long-tail data, a labeling error, a
domain mismatch, a corrupt item, or a mode the model has not learned.
Stratify by source and class, sample for human review, and cross-check
difficulty scores (error-vector norm, forgetting events, confidence and
variability across epochs) before deleting anything; the gradient-norm
score is unstable early in training.
