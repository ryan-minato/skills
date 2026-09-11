# Deep Diagnosis Tools

Read when the resident signals cannot separate the remaining causes and
a profiler, memory snapshot, communication debug window, or replay must
be switched on. Every tool here slows the run or floods storage: bound
the window, name the ranks, switch it off, and never benchmark with it
on. Verify option names against the framework's current documentation.

| Question | Tool | Window |
|---|---|---|
| where in the step does time go at operator or kernel level | the framework profiler with a schedule (skip, wait, warm-up, active) around T0; the system profiler's utilization, overlap, and straggler recipes | 5–20 steps |
| which allocation grew or fragmented | the allocator history with a bounded length, two snapshots diffed, the visualizer | from a stable point to the anomaly |
| which rank or collective misbehaves | the communication library's INFO level (topology, network, tuning) on named ranks; TRACE for a very short window; the framework's monitored barrier and consistency mode | seconds to a few steps |
| which node did the first non-finite value come from | per-rank non-finite hooks; full-health tensor dumps; replay on another device | one step |
| what the host is doing | a Python or system profiler over one step; host counters | one step |
| does the anomaly follow the device | replay of the same batch and checkpoint on another device | one step |

Keep profiler and snapshot outputs as artifacts in object storage with
their run id, step range, and rank, and link them from the evidence
chain; do not paste them into logs. Restore every debug setting to its
resident value when the window closes and record that it was restored.
