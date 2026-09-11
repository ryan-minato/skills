# Numerical Instability

Read when loss spiked, went NaN or Inf, diverged, or the precision
scaler keeps skipping steps.

## Timeline first

Assemble, around T0, per step: loss (mean and tail), learning rate and
schedule phase, precision scale and skipped steps, global and per-layer
gradient norm, clipping events, update magnitude (update-to-weight
ratio), optimizer second-moment health (`g²/v`, preconditioned update
RMS), activation and attention-logit statistics, and the batch identity.
A typical chain:

```text
schedule change or abnormal batch
→ one layer's gradient norm several × its rolling median
→ clipping rate jumps; update ratio jumps in that layer
→ precision overflow; scaler backs off or skips
→ loss non-finite
```

The first link in the chain is the cause; the last is the alert.

## Ranked causes

1. **Learning rate or schedule**: warm-up end, a resume with a
   misaligned scheduler, an accumulation change without a rate change.
2. **An abnormal batch**: a sample with an extreme loss, a corrupt or
   mis-tokenized example, a length outlier; visible in the per-example
   tail before the mean moves.
3. **Gradient or update blow-up along depth**: the depth ratio of layer
   gradient norms, residual-stream growth, attention logits growing.
4. **Optimizer state lag**: `v` far below the current `g²`, so the
   preconditioned update explodes; also a checkpoint restored without
   its optimizer state.
5. **Precision**: half-precision overflow in logits, attention scores, or
   a reduction; an `ε` too small for the dtype; an illegal operation
   (`log(0)`, division by a masked-out zero).
6. **Distributed or hardware**: one rank produces the anomaly; silent
   corruption that follows the device.

## Replay

Save the offending batch, the model, optimizer, and scaler states, and
the seed. Replay step T0: with the same batch and seed; in full
precision; on another device; with per-layer gradient, update, and
activation logging; and split into micro-batches and per-example
losses. Read the outcome:

| Reproduces | Points at |
|---|---|
| everywhere, in full precision | data (a sample or the batch composition), the model, or the optimizer state |
| only in reduced precision | overflow or underflow; find the first non-finite node |
| only on the original device | the device; quarantine and health-test it, recompute elsewhere |
| only with the restored optimizer state | a state mismatch at resume |

Locate the first non-finite node with hooks on loss, logits, activations,
gradients, optimizer states, and parameters (the frameworks' numeric
checks and full-health debug modes exist for this, at a cost).

## Fixing

Fix the first link: filter or reweight the offending data, repair the
schedule, restore the optimizer state, move the fragile computation to
full precision, clip per unit, or apply update clipping in the
optimizer. Lowering the global learning rate treats the symptom.
Absolute gradient thresholds do not transfer between models; judge
against the model's own rolling statistics. Bounded oscillation at a
large learning rate can be normal; runaway growth is not.
