# Hot-Path Performance and Readability

Read when readability and performance conflict in the step, data, or
kernel path.

## The rule

Readability wins wherever it is free: names, file organization, control
flow, formatting. In the training and inference hot paths — the step, the
data pipeline, custom kernels, collective communication — a meaningful
performance loss for form is not acceptable, because throughput and
memory can decide whether the research is feasible at all.

## Recovering understandability

When the fast implementation is hard to read, do not slow it down; add
around it:

1. **Local encapsulation**: the fused or hand-optimized code lives in one
   function or module with a plain interface; callers read the interface.
2. **A block comment** at the top: what the code computes, why it is
   written this way, and which measurement justified it.
3. **A slow reference implementation**: a clear, unoptimized version of
   the same computation, kept beside the fast one. It is documentation,
   a behavior specification, and a numerical oracle in one.
4. **A behavior test** comparing fast and reference outputs on small
   inputs, with the tolerance for the precision in use stated.
5. **A benchmark** that reproduces the speed-up on the stated hardware,
   so a later "simplification" can be measured before it lands.

## Deciding a request to simplify

When a reviewer asks to unfuse or restructure a hot-path computation for
readability: measure the cost on the project's hardware; if it is
meaningful, keep the fast path and add the items above; if it is not,
simplify. "Meaningful" is the project's call, recorded once in its
conventions (a percentage of step time or memory), not re-argued per
review.
