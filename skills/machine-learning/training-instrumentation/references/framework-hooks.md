# Framework Hooks

Read when wiring a specific framework's hooks: the PyTorch profiler,
memory snapshot, or component logging, the TensorFlow profiler or
debugger, or the JAX profiler, transfer guard, or NaN debugging. Verify
every option against the framework's current documentation; names and
defaults move between releases.

## PyTorch

- **Profiler**: records CPU and accelerator activity, operator shapes,
  tensor memory, and stacks, and exports a trace file. Never trace a
  long run end to end; use the schedule (`skip_first`, `wait`, `warmup`,
  `active`, `repeat`) for periodic windows, and an incident trigger for
  the rest.
- **Memory snapshot**: `torch.cuda.memory._record_memory_history()` plus
  a snapshot dump records allocator state, allocation and free events,
  segments, and OOM events with stacks. It sees only the framework
  allocator's memory — communication-library buffers and other native
  allocations are invisible — so compare with the device manager's total.
  Each record can be kilobytes; bound the history length on long runs.
- **Distributed debugging**: `TORCH_DISTRIBUTED_DEBUG=INFO|DETAIL`,
  `monitored_barrier()`, and the debug server. DETAIL adds consistency
  and synchronization checks and slows training; use it in a diagnosis
  window, never for benchmarks.
- **Component logging**: `TORCH_LOGS` (or the logging API) for autograd,
  the distributed stack, the data-parallel wrappers, the compiler stack;
  prefer it to source-level prints.
- **Communication library**: `NCCL_DEBUG=WARN` resident, `INFO` in an
  anomaly window (topology, network, tuning), `TRACE` only for a short
  window on named ranks; filter by subsystem (`NCCL_DEBUG_SUBSYS`).

## TensorFlow

- **Profiler**: overview (compilation, input, kernel launch, host
  compute, device compute, device-to-device), trace viewer, operator and
  kernel statistics, memory profile (allocation, deallocation,
  fragmentation, requested versus actual, per operator and step), input
  pipeline analyzer. Capture a bounded number of steps, remotely in
  distributed setups.
- **Debugger V2**: dump debug info in `FULL_HEALTH` mode (counts of
  negative, zero, positive finite, ±Inf, NaN per tensor) to locate the
  first non-finite event; `enable_check_numerics()` for a cheaper
  fail-fast.

## JAX

- **Profiler (XProf)**: per-device timeline, operator statistics,
  memory; view through TensorBoard or Perfetto. Typical patterns: device
  timeline gaps (host dispatch, data, logging, synchronization), a long
  first step (compilation), many tiny kernels, long collectives, memory
  pressure. Asynchronous dispatch makes naive wall-clock timing wrong:
  call `block_until_ready()` at benchmark boundaries.
- **Device-memory profiler**: snapshots at two points, diffed, locate
  growth caused by retained Python references.
- **Transfer guard**: watch explicit or implicit host-to-device,
  device-to-device, and device-to-host transfers.
- **`JAX_DEBUG_NANS`**: locates NaN sources at high cost; a short
  diagnosis switch, never resident.

## Devices and communication

- **Device manager exporter** (DCGM-class): utilization, memory-copy
  activity, memory, temperature, power, clocks, PCIe traffic, error
  codes, link bandwidth, tensor and memory activity, ECC and link
  errors, with pod and container labels on Kubernetes. Resident.
- **System profiler** (Nsight-class): utilization gaps, memory, MPI and
  device heat maps, communication overlap and straggler recipes, network
  and file-access analysis. Keep the profile as an artifact with summary
  metrics; do not convert it to text logs.
