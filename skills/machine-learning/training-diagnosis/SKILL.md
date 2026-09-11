---
name: training-diagnosis
description: >-
  Diagnoses a training run that misbehaves — locates the first anomaly
  across signals, walks a top-down ladder from correctness to data, host,
  transfer, compilation, kernels, memory, communication, storage, and
  hardware, confirms with a replay, and reports an evidence chain. Use
  when a run failed, diverged, slowed, stalled, ran out of memory, or is
  uneven across devices and the question is why or how to fix it: "GPU
  utilization fell to 40% mid-run", "loss went NaN at step 48120",
  "rank 3 is always slower", "OOM after two hours", "checkpoints take 20
  minutes", "NCCL timeout on every rank"; or when a loss spike tempts a
  blind learning-rate change. Not for deciding what to log or alert on
  before an incident, or for an environment or installation problem
  unrelated to a running training job.
license: Apache-2.0
compatibility: The bundled script requires Python 3.10+ (standard library only).
---

# Training Diagnosis

Precedence on every incident: an explicit user instruction, then the
project's own conventions and constraints, then this skill's defaults,
then tool preferences. A symptom is not a cause: low utilization, a
timeout, or a NaN is where the investigation starts, never where it ends.

## Method

1. **Freeze the evidence.** For a correctness failure (non-finite values,
   OOM, lost rank, timeout) keep the window before and after: the
   checkpoint, the offending batch identity, stacks, traces, platform
   events. A blind restart destroys the case.
2. **Find T0.** Establish the earliest step or timestamp at which any
   signal left its baseline — not the last error. Align signals by
   global step and by wall clock (clocks synchronized; step ids are the
   fallback join), across ranks. Check what changed at T0: a scheduler
   phase, a data shard, a checkpoint, a configuration or platform change
   from the audit trail. [`scripts/scan_series.py`](scripts/scan_series.py)
   finds the first sustained deviation in a metric series and, with a
   grouping column, the slowest-to-median ratio per step:

   ```bash
   python3 scripts/scan_series.py --input series.csv --column step_time_s --group-by rank
   ```

   It prints JSON and exits 0; `--help` documents the options and the
   exit codes.
3. **Walk the ladder top-down**, naming what each rung rules out:
   - Is correctness damaged? (non-finite, divergence, lost rank) → the
     numerical and distributed branches first.
   - Compare with the healthy baseline: step time, throughput, device
     utilization, memory, communication fraction, data wait.
   - Device timeline gaps? → split the wait into data, host work,
     host-to-device transfer, compilation, synchronization.
   - No gaps? → kernels and memory: compute-bound versus
     memory-bandwidth-bound, allocator state, fragmentation.
   - Then communication (per-rank collective time, skew, links), storage
     (latency, queues, control plane), and hardware (error codes,
     thermal, corrected errors).
4. **Confirm before fixing.** Reproduce with the smallest change that
   isolates the cause — a replay of the offending step, a single-rank
   run, a data-loader-only run, a storage microbenchmark — then apply the
   fix and re-measure against the baseline.
5. **Report the evidence chain** (below) and name the signal that was
   missing.

Done when: T0 is named, every rung that was ruled out has its evidence,
the cause reproduces, the fix is verified against the baseline, and the
chain is written.

## Symptom index

| Symptom | First three checks | Read |
|---|---|---|
| loss spike, NaN/Inf, divergence, scaler keeps skipping | first non-finite node; grad and update timeline; second-moment lag | [references/numeric-instability.md](references/numeric-instability.md) when loss spiked, went NaN or Inf, diverged, or the precision scaler keeps skipping steps |
| train and validation diverge; calibration or a class degrades; suspected shift | evaluation integrity (mode, preprocessing); per-class and tail losses; drift tests | [references/generalization-and-data.md](references/generalization-and-data.md) when train and validation diverge, calibration or per-class metrics degrade, or an input-distribution shift is suspected |
| device idle, step time up, throughput down | data-wait fraction; host CPU and transfer; compilation and synchronization | [references/throughput-and-utilization.md](references/throughput-and-utilization.md) when the device is idle, step time regressed, or data loading, compilation, or host work is suspected |
| OOM, memory grows, reserved far above allocated | active versus reserved versus device total; allocation history; batch shape tail | [references/memory.md](references/memory.md) when a run hits an out-of-memory error, memory grows over time, or reserved memory far exceeds allocated |
| ranks uneven, collective timeout, fabric errors, device errors | slowest/median ratio; the slow rank's node, device, link; earliest single-rank anomaly | [references/distributed-and-hardware.md](references/distributed-and-hardware.md) when ranks disagree in speed, a collective times out, communication or fabric errors appear, or hardware health events fire |
| checkpoint or data read slow, storage alerts | storage latency versus link throughput; request concurrency and queues; metadata service | [references/checkpoint-and-storage.md](references/checkpoint-and-storage.md) when checkpoints or data reads are slow, or storage latency alerts fire |
| resident signals cannot separate the remaining causes | profiler window; memory snapshot; communication debug window; replay | [references/deep-diagnosis-tools.md](references/deep-diagnosis-tools.md) when the resident signals cannot separate the remaining causes and a profiler, memory snapshot, communication debug window, or replay must be switched on |

## Numerical instability and replay

Reconstruct the timeline across loss, learning rate, precision scale,
gradient norm and clipping, update magnitude, optimizer second moment,
activations, and the batch identity. The usual order of suspicion:
learning-rate or schedule change → an abnormal batch → gradient or update
blow-up → optimizer second-moment lag → precision overflow → distributed
or hardware fault. Then replay:

```text
save the offending batch, model state, optimizer state, scaler state
→ replay step T0 with the same batch and seed
→ in full precision
→ on another device or node
→ with per-layer gradient, update, and activation logging
→ split into micro-batches and per-example losses
```

Reproduces everywhere → data or model and optimizer state, and hardware
is ruled out (a hardware cause is plausible only when the anomaly follows
the device); isolate the micro-batch or layer. Reproduces only on the
original device → quarantine that device for a health test and recompute
the affected steps elsewhere.
Do not lower the learning rate and restart before the replay: it hides
the cause and the same spike returns later.

## Deep diagnosis

Resident signals answer most incidents. Switch a deep instrument on for
a bounded window and switch it off again: a profiler schedule around T0,
an allocator history with a bounded length, a communication debug level
on the named ranks, the framework's distributed consistency checks. Each
slows the run and floods storage when left on. Never benchmark with a
debug mode enabled.

## The evidence chain

Deliver every diagnosis as the chain in
[`assets/evidence-chain.md`](assets/evidence-chain.md): T0; the
symptom; the signals consulted with their values; the causes ruled out
and why; the root cause; the fix; the verification that confirmed it;
the signal that was missing. Keep the chain to what the evidence
supports; when two causes remain, report both with the deciding signal to
capture next, and do not pick one. Run a sensitivity check on any log
excerpt before it leaves the machine. This skill pairs with
`sensitivity-check` for it. If it is not installed, load the
`ryan-minato-skills-installing` skill and install `sensitivity-check` as
it directs; never run an install command yourself. (If that installer
skill is absent too, it lives in the `core` catalog of
https://github.com/ryan-minato/skills.) If the user declines, review the
excerpt yourself for credentials, personal data, and raw samples, and
say so.

## Handoffs

- A missing signal that would have decided the case belongs to the
  instrumentation role. This skill pairs with `training-instrumentation`
  for it. If it is not installed, load the `ryan-minato-skills-installing`
  skill and install `training-instrumentation` as it directs; never run
  an install command yourself. (If that installer skill is absent too, it
  lives in the `core` catalog of https://github.com/ryan-minato/skills.)
  If the user declines, name the missing signal and its frequency in the
  evidence chain and add nothing else.
- Comparing two runs to see what differed is the run-identity role. This
  skill pairs with `experiment-provenance` for it. If it is not installed,
  load the `ryan-minato-skills-installing` skill and install
  `experiment-provenance` as it directs; never run an install command
  yourself. If the user declines, compare the commit, the resolved
  configuration, the environment identity, and the input identities by
  hand.

## Gotchas

- A collective timeout on every rank is usually the consequence of one
  rank's earlier crash, hang, or hardware event; find that rank and its
  T0 before blaming the fabric.
- Utilization at 100% is not efficiency: tiny kernels, memory-bound
  kernels, and spin-waits all read busy.
- An idle network link does not clear the storage path: request-queue
  saturation and metadata limits throttle writes while bandwidth sits
  unused.
- The framework's allocator statistics see only their own allocations;
  communication buffers and other native allocations are invisible.
- Bounded oscillation at a large learning rate can be normal training at
  the edge of stability; runaway growth is the failure.
- A seed does not make a run deterministic; a "same seed, different
  result" comparison proves nothing without the determinism flags.
- The last error message is the end of the causal chain, not its start.
