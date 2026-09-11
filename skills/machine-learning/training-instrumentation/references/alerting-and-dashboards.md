# Alerting and Dashboards

Read when the request includes alerts, severity, on-call routing, or
dashboards.

## The alert set

Start with these; add a rule only when a symptom escaped them. Each
condition is relative to the run's healthy baseline unless it is a
non-finite or hard-failure event.

| Alert | Level | Condition (starting values) |
|---|---|---|
| TrainingNoProgress | CRITICAL | no step completed for max(3× p99 step time, the business tolerance window) |
| TrainingNumericalFailure | CRITICAL | non-finite loss, parameter, or gradient that the run does not recover from automatically |
| DeviceOOM | CRITICAL | unrecovered accelerator out-of-memory |
| DistributedTimeout | CRITICAL | collective or heartbeat timeout |
| CheckpointUnavailable | CRITICAL | no valid restore point can be written and the risk window is unacceptable |
| JobUnexpectedExit | ERROR → CRITICAL | non-zero exit; level by automatic-recovery capability |
| ThroughputRegression | WARNING → ERROR | p95 step time > 1.2× / 1.5× baseline, persistent; ERROR when throughput also falls > 30% |
| DataPipelineStall | WARNING → ERROR | data-wait fraction high and throughput falling |
| MemoryHeadroomLow | WARNING | peak device memory > 85% and rising |
| RankStraggler | WARNING → ERROR | slowest rank / median > 1.2× / 1.5×, persistent |
| CommunicationRegression | WARNING | communication fraction or latency clearly above baseline |
| StorageLatencyRegression | WARNING | read, write, or checkpoint latency clearly worse than baseline |
| HardwareHealthError | ERROR → CRITICAL | device error codes, uncorrectable memory errors, link faults; level by impact |
| TelemetryBlindSpot | ERROR | a critical agent, scrape, or collector signal disappeared |

## Rule shape

- **Robust deviation**: `z = (x − median(window)) / (1.4826 · MAD(window) + ε)`
  and the relative ratio `r = x / (median(window) + ε)`, computed over
  a trailing window of the same run (or the healthy baseline run for
  the first minutes).
- **Persistence**: k of the last m evaluations beyond the threshold
  before the alert fires; a single spike is a dashboard event.
- **Composite**: `Instability = LossSpike ∧ (GradSpike ∨ UpdateSpike ∨ NonFinite)`;
  `DataAnomaly = LossTailSpike ∧ (SourceShift ∨ LabelShift ∨ FeatureDrift)`;
  `SystemSlowdown = ThroughputDrop ∧ WorkerDispersion`.
- **Completion-time SLO** for long runs: page when the current speed
  would miss the agreed finish time, not when a device dips for a
  minute.
- Starting numbers: `|z| > 4` for 3 evaluations → WARNING; `|z| > 6` or
  a composite → ERROR; non-finite → CRITICAL on first occurrence.

## Severity by consequence

The same source produces different levels: device telemetry is INFO
resident, WARNING on sustained throttling, ERROR on repeated error codes,
CRITICAL when a reset ends the job. Write the level into the rule, not
into the source.

## Dashboards

One platform dashboard with fixed rows beats a different view per team:

| Row | Panels |
|---|---|
| Alive | train/validation loss, learning rate, step p50/p95/p99, tokens or samples per second, estimated finish |
| Optimization | global and per-layer gradient norm, update-to-weight ratio, clipping rate, optimizer second-moment health, precision scaler and skipped steps |
| Internals | activation RMS and quantiles, gradient histograms (low frequency), weight norms, normalization statistics |
| Generalization and data | generalization gap, calibration, per-class results, validation drift |
| Compute and memory | device utilization, tensor-core and memory-bandwidth activity, power, clocks; device active/reserved/total memory, host RSS, OOM and retry events |
| Data and storage | data wait, queue depth, storage latency and throughput, checkpoint duration |
| Distributed | communication fraction, collective latency, per-rank step-time heat map, fabric errors |
| Platform | job and pod state, restarts, node events, collector health |

Link from a histogram bucket or an alert to the trace or profile of that
window (exemplars), so a p99 spike opens the evidence in one step.

## Model-health alert levels

| Level | Example condition | Action |
|---|---|---|
| INFO | validation plateau; clipping rate slightly up | record |
| WARNING | loss `z > 4` persistent; throughput −15%; a layer's update ratio above its p99 | raise sampling; save a checkpoint |
| ERROR | loss and gradient spike together; clipping rate high for long; consecutive scaler skips | save the offending batch and state; consider pausing |
| CRITICAL | any non-finite parameter or loss; unrecovered OOM; divergence; worker correctness mismatch | stop updates; roll back to the last healthy checkpoint |
| HARDWARE | the anomaly reproduces only on one device with the same batch and checkpoint | quarantine the node; replay elsewhere |
