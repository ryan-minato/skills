# Tests, Docstrings, and Typing for Tensor Code

Read when writing tests or docstrings for tensor code, or when a type
checker is proposed for it.

## What to test

The project's own contracts, where a mistake would corrupt a result:

- shape contracts across the modules the project wrote (batch, sequence,
  feature order; broadcasting assumptions);
- mask semantics (which value means "attend", padding positions excluded
  from losses and metrics);
- label alignment (shifted targets, ignore indices, class mapping);
- reduction semantics (mean over tokens versus over sequences; how
  padding affects the denominator);
- custom layers, losses, samplers, data filters, and metric
  implementations — against small hand-computed cases or a slow reference
  implementation;
- adaptation and integration code around a third-party library — not
  the library itself.

Use small real tensors on the CPU. Mocking tensor operations tests the
mock.

## Suites and cost

| Suite | Runs | Requirement |
|---|---|---|
| default | every check, ordinary CI | light, CPU-compatible, seconds to a few minutes |
| gpu | explicit command only | GPU present; **fails** without it (no skip, no fallback) |
| slow / heavy | by hand | anything needing large data, downloads, long runs, or a full training equivalence |

GPU requirement and cost are independent axes: a test can be GPU-only and
light, or CPU-only and expensive. Mark and route by both. Git hooks run
the formatter and the linter at most.

## Docstrings and comments

For a tensor-heavy interface, the docstring or a nearby comment states
what a Python type cannot: shape with named dimensions, dtype, device
expectations, layout, the mask convention, and the return semantics.

```python
def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of `x` over positions where `mask` is true.

    x: (B, T, D) float; mask: (B, T) bool, True = keep. Returns (B, D) in
    x.dtype; a row with no kept position returns zeros, not NaN.
    """
```

Block-level comments explain why: the mathematical intent, a numerical
reason (why the log is taken before the sum), a layout trick, distributed
behavior, a performance rationale. A comment that paraphrases the line
below it is noise.

## Typing

Annotations that aid reading are welcome; a global static type gate over
tensor code is not the default. The checker cannot verify shapes,
dtypes, devices, or masks, and making it pass costs casts, protocols,
and ignores without making the experiment more credible. Offer static
checking for configuration schemas, metadata, and control-plane modules,
where it pays.
