# Checkpoint and Storage

Read when checkpoints or data reads are slow, or storage latency alerts
fire.

## Bandwidth is rarely the limit

A checkpoint that takes many times its baseline while the network link
shows low throughput points at the storage **control plane**: request
concurrency limits (a fixed number of outstanding requests per client),
queueing, metadata operations, lock contention, small-object overhead,
or a throttled tenant — not the wire. Chain: concurrent writers →
request slots saturate → queues grow → application latency rises →
bandwidth stays idle → wall time explodes.

## Deciding signals

| Signal | Says |
|---|---|
| storage p95 latency versus link throughput | high latency with idle link → control plane |
| outstanding requests, queue depth, retries per client | saturation and back-off |
| metadata operation counts and latency | many small files or listings |
| checkpoint bytes, duration, and parallelism per rank | who writes, how much, how concurrently |
| data-loader queue depth and fetch latency | reads starving the run |

## Ranked causes

1. Client-side concurrency limits or too many parallel writers.
2. Metadata-heavy layouts (many small files, per-step listings).
3. A shared filesystem's server-side limits or a throttled bucket.
4. Serialization on the host (pickling, compression) before the write.
5. A degraded link (then the distributed branch).

## Fixing

Reduce concurrent writers or raise the client's request limits,
consolidate small files, write asynchronously and off the training
critical path, stage to local disk then upload, compress or shard
deliberately, and verify with a storage microbenchmark that isolates the
change. Checkpoint retries are ERROR events; a run with no valid restore
point is CRITICAL.
