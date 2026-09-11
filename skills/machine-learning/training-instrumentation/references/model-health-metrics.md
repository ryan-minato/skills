# Model-Health Metrics

Read when instrumenting optimization dynamics or network internals beyond
loss and gradient norm, or when asked what a specific health metric
means. Each entry: definition, what normal looks like, sampling, and how
to alert. No single metric is a health score; diagnosis combines them.

## Objective layer

- **Loss and its distribution**: batch mean plus per-example p50/p90/p99/max;
  trend as an EMA difference; smoothness as a windowed coefficient of
  variation. Normal: long-horizon decrease with short-horizon noise;
  bounded oscillation at a large learning rate is not failure (training
  can sit at the edge of stability and still progress). Alert: robust-z
  of loss and of its slope; a tail (p99/median) that grows while the
  mean holds.
- **Generalization gap**: `L_val − L_train`, `Acc_train − Acc_val`. The
  sign is not diagnostic with strong augmentation or different
  train/eval paths; the trajectory is. Alert on validation degradation
  beyond its own noise relative to the best point, never on a fixed gap.
- **Learning rate and its bookkeeping**: the scheduled value, the
  optimizer step count, the global step, accumulation steps, effective
  batch size, and precision-scaler skipped steps. Any unplanned
  discontinuity (resume, accumulation change without a rate change,
  scheduler stepping while the optimizer skipped) is WARNING; a resume
  discontinuity is CRITICAL.

## Optimization dynamics

- **Gradient norm**: global L2 every step; per-layer norms as a
  depth-by-time heat map at low frequency; the depth ratio
  `max_l / min_l` for vanishing or exploding patterns. Alert relative to
  the post-warm-up rolling median, never on an absolute number across
  models.
- **Clipping rate**: fraction of steps clipped over a window. Clipping
  is not a defect; a rate persistently above roughly 20–50% means the
  optimizer keeps trying to take larger steps than allowed — investigate.
- **Update-to-weight ratio** `‖Δθ_l‖ / ‖θ_l‖` per layer, measured after
  the optimizer step from the previous parameters. More comparable than
  the raw gradient under adaptive optimizers. Alert on a layer that is an
  order of magnitude off its neighbors or its own history.
- **Gradient-to-weight ratio** `‖g_l‖ / ‖θ_l‖`: the "force over mass"
  view; adaptive gradient clipping works on this per unit.
- **Optimizer second-moment health**: for adaptive optimizers, log the
  mean of `v`, the ratio `g² / v` (values well above 1 mean `v` lags a
  growing gradient), and the RMS of the preconditioned update
  `g / (√v + ε)`. Underestimation of `v` has been observed steps before
  loss spikes in large low-precision runs; this is the early warning.
- **Gradient variance and noise scale**: split a batch into 2–8
  micro-batches, log their gradient norms and pairwise cosine; a rising
  noise scale argues for a larger batch, a suspiciously low one for
  duplicated data or a sampler bug. Per-sample gradients only offline.
- **Gradient distribution**: mean, std, quantiles, max |g|, near-zero
  fraction on a sampled subset (0.1–1% of parameters or a fixed layer
  set) every 100–1000 steps; drift as a distance to a healthy snapshot.
- **Gradient alignment**: step-to-step cosine; task-to-task cosine in
  multi-task training. Negative alignment is a signal to inspect
  sampling, weights, and rates, not an automatic defect.
- **Weight norm, sparsity, stable rank** `‖W‖_F² / ‖W‖_2²`: norm jumps
  point at optimizer-state or checkpoint problems; stable rank collapse
  and neighboring-layer Jacobian alignment are research-level precursors
  computed at checkpoints, not alerts.
- **Sharpness** (top Hessian eigenvalue by power iteration on
  Hessian-vector products): for analysis at checkpoints; not a health
  score, and its relation to generalization is not settled for large
  models.

## Network internals

- **Activation statistics** per sampled layer: mean, std, p1/p50/p99,
  max |a|, near-zero fraction, saturation fraction for bounded
  activations; for transformers, embedding RMS, residual-stream RMS per
  layer, attention-logit max and RMS. Record statistics only, never the
  tensors. Watch for sudden shifts, variance collapse, tail growth, and
  exponential growth or decay with depth.
- **Normalization statistics**: batch-versus-running mean distance
  normalized by the running standard deviation, and the variance ratio;
  drift points at small batches, wrong train/eval mode, domain shift, or
  distributed normalization configuration.

## Numerics

- **Non-finite fraction** hooked at loss, logits, activations,
  gradients, optimizer states, parameters, so the first non-finite node
  is known; the framework's numeric-check or full-health debugging modes
  do this at cost.
- **Precision scaler**: scale value, back-off events, skipped steps.
  Occasional back-off is normal; persistent collapse or frequent skips
  is ERROR.

## Data and generalization

- **Per-example loss spectrum** by source, class, language, length:
  two runs with the same mean can differ entirely in their tails.
- **Sample-difficulty scores** (error-vector norm, gradient norm,
  forgetting events, confidence and variability across epochs) identify
  candidates for review; a high score is not a deletion criterion — it
  marks hard, mislabeled, shifted, or corrupt examples alike, and the
  gradient-norm score is unstable early in training.
- **Calibration**: expected calibration error is bin-dependent and
  discontinuous; report it with a reliability diagram and a proper
  scoring rule (log loss or Brier), alert relative to the baseline,
  never against a universal cutoff.
- **Ranking and per-class metrics**: ROC-AUC, precision-recall AUC for
  imbalanced data, per-class recall; an overall accuracy that holds while
  one class collapses is the case these catch.
- **Validation drift**: KS or Wasserstein for numeric features,
  chi-square or Jensen–Shannon for categorical, PSI for histograms, MMD
  for embeddings. Library defaults (p < 0.05, distance ≥ 0.1) are tool
  defaults; data drift, concept drift, and performance degradation are
  three different questions.
- **Early stopping**: patience on a smoothed validation metric with
  restore-best; not on the raw noisy value when the set is small.

## Recording the Adam state (PyTorch)

```python
@torch.no_grad()
def adam_health(optimizer):
    v_mean, ratio, n = 0.0, 0.0, 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            v = optimizer.state.get(p, {}).get("exp_avg_sq")
            if v is None or p.grad is None:
                continue
            g2 = p.grad.float().pow(2).mean()
            v_mean += v.float().mean().item(); ratio += (g2 / (v.float().mean() + 1e-30)).item(); n += 1
    return {"adam/mean_v": v_mean / n, "adam/g2_over_v": ratio / n} if n else {}
```
