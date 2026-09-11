# Throughput and Utilization

Read when the device is idle, step time regressed, or data loading,
compilation, or host work is suspected.

## The device is the consumer, not the cause

"Utilization dropped to 40%" describes the end of a chain. Split the
step into stages (data, forward, loss, backward, optimizer,
synchronization) and ask which grew.

## Ranked causes

1. **Data starvation**: data-wait fraction up; queue depth at zero;
   storage latency up. Chain: storage latency ↑ → loader queue drains →
   late batches → device gaps → utilization ↓ → step time ↑. Confirm
   with a loader-only run and a storage read benchmark.
2. **Host work**: CPU saturated by decoding, augmentation, tokenization,
   Python overhead, logging, or garbage collection; process RSS growth.
3. **Host-to-device transfer and synchronization**: unpinned memory,
   unintended device-to-host syncs (`.item()`, `.cpu()`, prints of
   tensors), implicit syncs in metrics every step.
4. **Compilation and recompilation**: a slow first step is expected;
   repeated recompiles from dynamic shapes or graph breaks are not.
5. **Communication and stragglers**: communication fraction up or rank
   skew — the distributed branch.
6. **Kernels**: many tiny kernels, memory-bandwidth-bound kernels, a
   regression after a library or driver change; visible only in a
   profiler window.
7. **Thermal, power, and hardware**: clocks down, throttling, corrected
   errors.

## Deciding signals

| Candidate | Decides |
|---|---|
| data | data-wait fraction; loader queue depth; storage p95 |
| host | CPU %, RSS, Python profiler over one step |
| transfer/sync | trace: gaps between host launch and device execution; sync markers |
| compile | compile events, graph-break counters, recompile counts |
| kernels | profiler window: kernel durations, tensor-core and memory activity |
| hardware | clocks, temperature, power, error counters |

Use the profiler in a bounded window only after the stage split points
at the kernel level. For asynchronous frameworks, wall-clock timing
without blocking on results measures dispatch, not execution.
