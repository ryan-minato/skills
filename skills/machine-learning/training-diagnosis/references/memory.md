# Memory

Read when a run hits an out-of-memory error, memory grows over time, or
reserved memory far exceeds allocated.

## Three views, joined

```text
device manager: total used on the device
framework allocator: active (in use by tensors), reserved (cached by the allocator), peak
allocator history: allocation and free events with stacks
```

Compare them across time (the step of OOM and thousands of steps
before):

| Pattern | Diagnosis |
|---|---|
| active rises monotonically across epochs, reserved tracks it | a **leak**: tensors or Python references retained (a list of losses with graphs attached, metrics kept on device, a retained computation graph) |
| active flat, reserved far above it with many small free blocks | **fragmentation**: allocator caching and size-class churn; variable shapes, checkpoint-reload cycles |
| framework total well below the device manager's total | **non-framework allocation**: communication-library buffers, kernels' workspaces, another process, a native extension |
| a sudden OOM on one batch with a stable baseline | a **shape outlier**: sequence or image length tail, an unpadded batch |
| memory differs strongly between ranks | uneven sharding, an asymmetric parallelism layout, one rank holding evaluation state |

## Confirming

- Two allocator snapshots (a stable point and a later one) diffed show
  what grew and from which stack.
- A leak survives `del` and empty-cache; fragmentation does not.
- The device manager's total minus the framework's reserved is the
  non-framework share.
- Replay the OOM batch alone to confirm a shape outlier.

## Fixing

Leaks: detach or move metrics off device, drop retained graphs, bound
histories. Fragmentation: stabilize shapes (bucketing, padding), tune
the allocator's configuration, reduce reload churn. Non-framework: size
communication buffers deliberately, account for workspaces, isolate
processes. Outliers: cap lengths or bucket batches. Headroom alerts
follow trend and retries, not a single percentage.
