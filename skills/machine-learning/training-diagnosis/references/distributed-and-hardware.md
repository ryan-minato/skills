# Distributed Training and Hardware

Read when ranks disagree in speed, a collective times out, communication
or fabric errors appear, or hardware health events fire.

## Synchronous training runs at the speed of the slowest rank

Compute, per step, `skew = max_r T_r / median_r T_r` for step time and
for collective time. All ranks slow together → congestion, a global
change, or a shared dependency (storage, a parameter server). A few
ranks slow → their node, device, and link.

## Ranked causes for a straggler

1. One device throttled, erroring, or degraded (clocks, temperature,
   corrected memory errors, error codes).
2. One node's link degraded (retransmits, replay errors, a down lane).
3. Uneven work: data imbalance, a longer sequence bucket on one rank,
   an expert overloaded in a mixture-of-experts run, a pipeline bubble
   misread as idleness.
4. Host contention on that node (another process, CPU, I/O).
5. Fabric congestion affecting a subset.

## Timeouts

A collective timeout on every rank is a consequence: find the rank
whose signals left baseline first — a crash, a data-loader hang, an OOM,
a hardware event — with per-rank step time, heartbeats, and the logs
before the timeout. Only when every rank was healthy until the same
moment is the fabric the first suspect. The framework's monitored
barrier reports which ranks failed to arrive; the debug consistency mode
verifies collective shapes and order, at a cost, in a window.

## Fabric and links

Per-link throughput, retransmits, replay errors, link state, and the
device manager's link-error counters, correlated by node and rank.
Compare against the healthy baseline of the same link; "the NIC is not
saturated" says nothing about a degraded link's latency.

## Hardware and silent corruption

Error codes, uncorrectable memory errors, resets, thermal events, and
repeated corrected errors mark a device for quarantine. Silent data
corruption shows as anomalies that follow the device: replay the same
batch and checkpoint on another device; if the anomaly stays with the
original device, quarantine it and recompute the affected steps.

## Parallelism specifics

Record the pipeline stage and micro-batch id, otherwise bubbles read as
stalls; in tensor parallelism watch small-collective latency and
overlap; in mixture-of-experts the hot link is often a routing imbalance
(expert load, dropped tokens, all-to-all volume) rather than a fault.
