---
name: training-instrumentation
description: >-
  Designs what a training run emits and how it is judged — metrics,
  stage-level traces, and scheduled profiles across the outcome,
  optimization, and systems layers with shared correlation keys, a minimal
  signal set by priority with sampling frequencies, model-health metrics
  beyond loss, and symptom-first alerts relative to a healthy baseline
  under five severities. Use when deciding what a training loop or job
  should log or measure and how often — "we only print loss", "what should
  a serious loop record", "monitor a 64-GPU job"; when setting up alerts,
  severities, on-call paging, or dashboards for training; when asked
  whether a metric's value (a gradient norm, a data-wait fraction, a
  memory level) is normal; or when wiring a profiler, memory snapshot, or
  communication debug level. Not for diagnosing a run that already failed
  or slowed, or for logging in a service that trains nothing.
license: Apache-2.0
---

# Training Instrumentation

Precedence on every run: an explicit user instruction, then the
project's own conventions and constraints, then this skill's defaults,
then tool preferences. Every number below is an initial value to
recalibrate against the project's own healthy baseline.

## Principles

- **Alert on symptoms, diagnose causes.** Page for "training makes no
  progress", "throughput fell", "numbers are no longer finite", "the job
  is about to fail" — not for every cause that could produce them. A
  utilization dip is a diagnostic signal, not a page.
- **Resident and deep layers.** Cheap signals run always; operator
  traces, allocator histories, full communication traces, and profilers
  run in scheduled windows or on an incident trigger. An always-on full
  profiler is refused: its output is enormous and it slows the run.
- **Three layers.** Outcome (is the model better: validation and
  benchmark metrics), optimization (is training healthy: loss, gradients,
  updates, activations, precision), systems (is it running efficiently:
  step time, data wait, throughput, memory, utilization, communication).
  A small single-process project may thin the systems layer; it never
  drops the other two.
- **Three instruments, separated.** Metrics are continuous and cheap;
  traces show execution structure at stage level (data, forward, loss,
  backward, optimizer) and no finer by default; profiling is the
  operator- and kernel-level instrument, switched on for a window when a
  stage is anomalous.

## Correlation keys

Every emitted record — metric, log line, span, profile — carries the
keys that let signals join across layers:

| Key | Purpose |
|---|---|
| `run_id`, `job_id` | which execution |
| `global_step` (and micro-batch where used) | align every layer on training progress |
| `rank`, `world_size`, `local_rank` | which process; straggler detection |
| `node`/`pod`, `device_id` | which hardware; join with device and platform telemetry |
| `phase` | data, forward, loss, backward, optimizer, eval, checkpoint |
| `trace_id`, `span_id` | join a log or metric with a trace when traces exist |
| timestamp | wall-clock join; clocks must be synchronized across nodes, and step ids are the fallback join |

Structured log records follow the OpenTelemetry log model (timestamp,
severity, body, resource attributes such as run and rank, record
attributes such as step and phase) so every source shares one envelope.

**Cardinality rule.** Unbounded values — step, sample id, file path,
tensor name, stack trace, checkpoint URI, user — never become metric
labels; each label combination is a separate series. Step goes on the
value axis; ids go in log fields.

## The minimal set

Build in priority order; each level makes the next kind of question
answerable.

| Priority | Question | Signals |
|---|---|---|
| P0 — failures explainable | why did it stop? | run identity; job state and exit reason; uncaught exceptions; device OOM and non-finite events; accelerator health (errors, thermal, resets); platform events (eviction, host OOM, mount failures); checkpoint save/load results; telemetry self-health |
| P1 — regressions detectable | is it slower or worse? | loss, learning rate, validation metric; step time p50/p95/p99; samples or tokens per second; data-wait fraction; global gradient norm; device memory used/reserved/peak; per-rank step time and skew; communication time per step; storage latency |
| P2 — regressions locatable | where? | stage-level trace; per-layer gradient view; update-to-weight ratio; optimizer second-moment health; data pipeline breakdown (fetch, decode, collate, queue depth); host CPU and RSS; transfer time; scheduled profiler windows |
| P3 — automated correlation | what caused it? | incident-triggered profiles and snapshots; tail sampling of slow ranks; ring buffer of the window before a trigger; baseline comparison across runs |

For a single-device experiment the smallest useful set is P0's run
identity, exit reason, and non-finite events plus P1's loss, learning
rate, step time, throughput, global gradient norm, device memory, and
data-wait fraction at the catalog frequencies; per-rank, communication,
storage, and platform telemetry wait until the run spans more than one
node. Read [references/cluster-telemetry.md](references/cluster-telemetry.md)
when training spans more than one node, or when designing collection,
storage, sampling, retention, or capacity for training telemetry.

## Signal catalog

System and platform signals, with a starting frequency:

| Signal | Frequency | Notes |
|---|---|---|
| step time, throughput, loss, learning rate | every step (aggregate to seconds at cluster scale) | the training heartbeat |
| data wait, fetch/decode/collate time, queue depth | per step, aggregated | the first suspect for idle devices |
| device utilization, memory, temperature, power, clocks, errors | 1–5 s | from the device manager (DCGM/NVML-class exporters) |
| host CPU, RSS, threads, disk, network | 5–15 s | node or process exporter |
| communication: collective type, bytes, duration, per rank | per step summary; full trace on demand | comm fraction and rank skew derive from it |
| per-rank step time, barrier wait | per step | straggler ratio = slowest / median |
| storage read/write latency, throughput, errors, checkpoint duration | per event | control-plane queues matter as much as bandwidth |
| job and platform events, exit codes, restarts | per event | the scheduler and the container runtime, never polled aggressively |
| compile/JIT time, graph breaks, recompilations | per event | a slow first step and repeated recompiles |
| telemetry pipeline drops, scrape failures | continuous | silence is not health |

Model-health signals (the optimization layer), with a starting frequency:

| Signal | Frequency | Why |
|---|---|---|
| loss (mean and per-example p50/p90/p99/max) | every step; quantiles every 10–100 | a mean hides a heavy tail |
| global gradient norm, clipping rate | every step | already computed by clipping |
| per-layer gradient norm heat map | every 10–100 steps | where, not just whether |
| update-to-weight ratio (per layer) | every 10–100 steps | how far parameters actually moved after the optimizer |
| optimizer second-moment health (current g² versus v; preconditioned update RMS) | every 10–100 steps | v underestimation precedes spikes |
| non-finite counts; precision scaler value and skipped steps | every step | free; the first sign of overflow |
| activation statistics (RMS, quantiles, dead or saturated fraction) for sampled layers | every 100–1000 steps | signal propagation |
| histograms (gradients, weights) for a few layers | every 500–5000 steps | expensive; never every layer every step |
| validation metrics, calibration, per-class results, drift tests | per evaluation cycle | outcome and generalization |

Read [references/model-health-metrics.md](references/model-health-metrics.md)
when instrumenting optimization dynamics or network internals beyond loss
and gradient norm, or when asked what a specific health metric means.
Read [references/model-family-signals.md](references/model-family-signals.md)
when the model is a transformer language model, uses mixture-of-experts,
pipeline, or tensor parallelism, or trains with reinforcement-learning,
adversarial, or diffusion objectives. Read
[references/framework-hooks.md](references/framework-hooks.md) when
wiring a specific framework's hooks: the PyTorch profiler, memory
snapshot, or component logging, the TensorFlow profiler or debugger, or
the JAX profiler, transfer guard, or NaN debugging.

## Severity and alerts

Five severities, describing consequence rather than source:

| Level | Meaning | Action |
|---|---|---|
| CRITICAL | failed, about to fail unrecoverably, or results no longer trustworthy | page; stop or isolate; freeze the evidence window |
| ERROR | an operation failed; the run may continue by retry or fallback | high-priority ticket; trigger a deep-diagnosis window |
| WARNING | still correct, but performance, capacity, or stability at risk | dashboard and alert; raise local sampling |
| INFO | normal state and cheap summaries | resident |
| DEBUG | operator, kernel, stack, per-rank detail | off or sampled; incident-triggered |

Alert design:

- **Relative to a healthy baseline** of the same run or model: a ratio to
  the baseline percentile, or a robust z-score against a rolling median
  with median absolute deviation, plus a **persistence** rule (k of the
  last m windows) before paging. Absolute thresholds are starting values
  only: a gradient norm of 1.0 is normal for one model and optimizer and
  alarming for another.
- **Composite for instability**: a loss spike alone is a warning; a loss
  spike together with a gradient, update, or non-finite anomaly is an
  incident. Non-finite values in parameters or the loss need no
  statistics: first occurrence is CRITICAL.
- **Starting thresholds** (recalibrate): p95 step time > 1.2× baseline for
  5–10 min → WARNING, > 1.5× with throughput down > 30% → ERROR; data wait
  > 20% of step with device idle → WARNING, > 40% → ERROR; device memory
  > 85% and rising → WARNING, > 95% or allocation retries → ERROR;
  communication > 30% of step and worsening → WARNING, > 50% or clear
  rank skew → ERROR; slowest rank > 1.2× median → WARNING, > 1.5×
  sustained → ERROR; no step for longer than 3× the p99 step time →
  CRITICAL.
- **Metamonitoring**: the telemetry pipeline's own drops and missing
  sources alert, or silence reads as health.

Read [references/alerting-and-dashboards.md](references/alerting-and-dashboards.md)
when the request includes alerts, severity, on-call routing, or
dashboards.

## PyTorch skeleton

The order matters: backward, unscale, measure the true gradient, clip,
step, then measure the update from the previous parameters.
[`assets/torch_health.py`](assets/torch_health.py) is a drop-in module
that performs these measurements behind one function; the loop's single
logging seam receives its dictionary.

```python
health = TrainingHealth(model, max_norm=max_norm)  # once, before the loop
...
scaler.scale(loss).backward()
scaler.unscale_(optimizer)                      # measure real gradients
grad_norm = clip_grad_norm_(model.parameters(), max_norm)  # returns the pre-clip norm
scaler.step(optimizer); scaler.update()         # a skipped step means non-finite grads
metrics = health.after_step(model, optimizer, loss_vec, grad_norm, scaler)
log_metrics(step, metrics)                      # the one seam
```

`max_norm` is a project value, not a recommendation; pass the same value
to the helper or the clipping indicator stays at zero. Per-layer views,
the update-to-weight ratio, and bounded histograms run at their low
frequencies inside `after_step`; every step it reports the loss
quantiles, the gradient norm, finiteness of loss, gradients, and
parameters, and the scaler's scale and skipped steps.

## Privacy

Never in logs, metrics, traces, or the tracker: raw samples, prompts,
images, credentials, presigned URLs, user identifiers, full dataset
paths that carry tenant or personal information, tensor contents,
activation or gradient dumps, unredacted exception payloads. Log dataset
ids, sample hashes, object ids; route raw material to a governed
artifact store with access control and expiry; redact in the collector
before long-term storage.

## Handoffs

- Explaining a run that is already misbehaving is the diagnosis role.
  This skill pairs with `training-diagnosis` for it. If it is not
  installed, load the `ryan-minato-skills-installing` skill and install
  `training-diagnosis` as it directs; never run an install command
  yourself. (If that installer skill is absent too, it lives in the
  `core` catalog of https://github.com/ryan-minato/skills.) If the user
  declines, list the signals that would separate the candidate causes
  and stop there.

## Gotchas

- Device utilization is not efficiency: a device can read 100% busy while
  running tiny kernels, or idle behind a data loader. Pair it with step
  time, data wait, and the stage trace.
- A framework's allocator statistics see only its own allocations; the
  communication library's buffers and other native allocations are
  invisible, so compare with the device manager's total.
- Optimizer second-moment underestimation precedes loss spikes; logging
  only the gradient norm misses it.
- Calibration error is bin-dependent; report it with a reliability view
  and a proper scoring rule, never as a single truth.
- Drift-detection libraries' default thresholds are tool defaults, not
  physiological norms.
- Sharpness, stable rank, and Jacobian alignment are research signals,
  not health scores.
- The framework's distributed debug mode and the communication library's
  trace level slow training; they are diagnosis windows, never resident.
- Job accounting commands on a scheduler are not for polling every
  second; export once, never loop.
- Managed-platform profilers come and go; keep the model in open formats
  (OpenTelemetry, Prometheus, the framework's trace files).
