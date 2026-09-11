# Model-Family and Parallelism Signals

Read when the model is a transformer language model, uses
mixture-of-experts, pipeline, or tensor parallelism, or trains with
reinforcement-learning, adversarial, or diffusion objectives.

## Transformer language models

- Loss bucketed by token position, sequence length, language, source,
  domain, packing, and curriculum stage — not only the overall mean;
  perplexity is `exp(loss)` only under a consistent natural-log loss.
- Embedding RMS, residual-stream RMS per layer, attention-logit max and
  RMS, query and key norms; per-layer gradient and update ratios;
  optimizer second-moment health; precision-scaler skips; tokens per
  second.
- Loss spikes have several known mechanisms — second-moment lag,
  residual and Jacobian scale growth, stable-rank collapse, silent
  hardware corruption — so no single-rule "spike root cause" alert
  exists; the composite instability alert plus the replay procedure of
  the diagnosis role is the design.

## Data, tensor, and pipeline parallelism

- Data parallel: all-reduce or reduce-scatter time and bytes, per-rank
  step time, input balance across ranks.
- Tensor parallel: small-collective latency, high-speed link bandwidth
  and errors, kernel–communication overlap.
- Pipeline parallel: record `pipeline_stage`, micro-batch id, and bubble
  time; otherwise a stage's idle time reads as an anomaly.

## Mixture-of-experts

Expert load distribution, token routing entropy, all-to-all bytes and
duration, expert capacity and dropped tokens. A network hot spot in an
MoE run is as often a routing imbalance as a link fault; these signals
extend the training and communication catalogs rather than forming a
separate monitoring system.

## Reinforcement learning (policy optimization)

Episode return is the outcome-layer signal — there is no supervised
validation loss to propose — plus policy entropy, divergence between the
new and the old policy, the clipped fraction of probability ratios, value loss,
advantage mean and std, reward statistics. Divergence ceilings used to
stop an update early are implementation experience (small values such
as 0.01–0.05 in common implementations), not universal thresholds.
Typical patterns: return down with divergence up → the update is too
aggressive; entropy collapsing toward zero → premature policy collapse;
clipped fraction persistently high → most updates truncated; reward up
with divergence from the reference abnormal → inspect for reward
hacking.

## Adversarial training

Generator and discriminator losses, the discriminator's outputs on real
and generated samples, both sides' gradient norms and update ratios.
Theory's balance point is not a health threshold across objectives; a
side whose loss or gradient stays near zero while the other loses its
learning signal is the pattern to alert on.

## Diffusion

Loss bucketed by timestep or signal-to-noise band; a timestep-averaged
loss hides a degrading noise band.

## Silent data corruption (large-scale training)

Hardware faults have been observed to surface as transient loss and
gradient-norm spikes, attention-logit spikes, NaN propagation, and
persistent parameter divergence. Signals: the update-norm ratio to its
window median, attention-logit max, residual gain, and deterministic
replay or checksum comparisons across ranks. A single spike is not a
diagnosis; the replay on another device is.
