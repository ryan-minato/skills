# Cluster Telemetry: Collection, Sampling, Retention, Capacity

Read when training spans more than one node, or when designing
collection, storage, sampling, retention, or capacity for training
telemetry.

## Collection architecture

```text
trainer (scalars, step, grad)       → OpenTelemetry collector → log store / trace store
framework profiler (trace files)    → profile/object pipeline → object storage (metadata → logs)
communication library / runtime     → collector
device manager exporter (DCGM-class) ┐
node / process exporter              ├→ metrics pipeline (Prometheus-compatible) → time-series DB
storage / NIC telemetry              ┘
scheduler / platform events         → collector
                                     dashboards and alerting read all four stores
tracker / TensorBoard                ← the experiment view, beside the platform stores
```

Collector processing order: ingest → memory limit → early filter → parse
and normalize → resource enrichment (run, job, rank, node) → redaction →
sampling → batch → export. Redact before long-term storage, not at query
time. Keep the experiment tracker and TensorBoard as the experience layer;
store hardware errors, platform events, and raw distributed traces
independently so an outage of the tracker does not lose the incident
data.

## Per-platform mapping

- **Kubernetes**: container stdout via a node agent to an independent
  backend (pod deletion drops local logs); ingest events — OOMKilled,
  Evicted, FailedScheduling, FailedMount, image pulls, device-plugin and
  runtime events — because a CUDA OOM, a cgroup OOM, and an eviction all
  look like "the process vanished" from inside the trainer. Keep raw
  event fields; system log formats are not a stable API.
- **Slurm**: accounting (`sacct`-class data: state, exit code, max RSS,
  I/O, energy, allocated resources, restarts, failed node) exported into
  the log store by the accounting daemon, never polled in a loop from
  the training script.
- **Managed cloud training**: job stdout/stderr and per-instance streams
  land in the cloud's log service with metric extraction by pattern;
  managed profilers are product lifecycle risks, so keep the model in
  open formats.
- **Prometheus-class metrics**: recording rules precompute cross-rank
  aggregates; alert rules use `for` durations; label cardinality is
  bounded (cluster, namespace, job type, model family, device type,
  phase, collective type, error class are labels; run id, rank, node are
  finite per run but heavy over time; step, sample id, path, tensor
  name, stack, URI, user never).
- **Dashboards**: exemplars from a histogram bucket to a trace; log
  fields link to the trace id.

## Sampling tiers

| Signal | Small (1–8 devices) | Medium (16–256) | Large (512+) |
|---|---|---|---|
| training scalars | 1–10 steps | 5–20 steps | local 10–100 steps; cluster aggregate 5–15 s |
| device, node metrics | 5–10 s | 2–10 s | 5–15 s; 1 s on anomalous nodes |
| data and communication summary | per step | 1–10 steps | aggregated 5–15 s; keep the rank tail |
| full operator/kernel trace | frequent manual | periodic 5–20 steps | very low ratio; incident windows only |
| memory snapshot | on OOM or by hand | above threshold or OOM | anomaly-triggered |
| communication trace level | by hand | anomaly window | very short window, named ranks |
| ERROR/CRITICAL events | 100% | 100% | 100% — never probabilistically sampled |

Do not drop the slow rank: all ranks emit a cheap summary; the pipeline
computes p95/p99 and skew; the anomalous rank, step, and phase are
tail-sampled with full detail. Keep a rolling pre-trigger ring buffer so
an incident freezes the window before and after it — a timeout is often
the consequence of an earlier single-rank anomaly.

## Retention (starting values)

| Data | Hot | Warm / cold |
|---|---|---|
| CRITICAL/ERROR events | 90 days | 1 year or per compliance |
| training scalars and summaries | 90–180 days | aggregates 1–3 years |
| device and node metrics at 5–15 s | 30–90 days | 1–5 min downsample long term |
| INFO application logs | 14–30 days | key events 90 days+ |
| DEBUG and communication INFO | 1–7 days | frozen incident cases only |
| raw operator/kernel traces | 24 h–7 days | representative cases only |
| memory snapshots | 7–30 days | OOM cases 90 days+ |
| audit and security logs | 90 days+ | per policy; never shortened by the ML team |

## Capacity model

Logical bytes per day `D = Σ_i events/s_i × bytes/event_i × 86400`;
physical `D × replication / compression`. Cost is ingest + hot storage +
archive + indexing (cardinality) + query scan + egress + replication. A
planning assumption — not a measurement — of roughly 1.1 GB per device
per day (scalars and metrics ~0.25, structured logs ~0.35, node and
platform share ~0.15, data and communication summaries ~0.15, sampled
profiler artifacts ~0.2) gives about 35–280 GB/day for 32–256 devices and
1–4.5 TB/day for 1,024–4,096; the point is that the model is
parameterized, and resident full traces would invalidate it entirely.

## Telemetry self-health

Collector drops, export queue depth, scrape failures, remote-write
errors, and missing critical sources are alerted in their own domain:
a monitoring system that fails silently produces "no alerts, so all is
well".

## Privacy in the pipeline

Deny by default in logs: raw text, prompts, images, credentials,
presigned URLs, user ids, tenant-bearing paths, tensor contents,
activation or gradient dumps, unredacted exception payloads. Sampled raw
inputs for data debugging go to a separate governed store with access
control, expiry, and audit — never into the log store.
