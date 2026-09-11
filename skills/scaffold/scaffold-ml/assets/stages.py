"""The stage-trace seam: one context manager per training stage.

    with stage("data"):      batch = next(iterator)
    with stage("forward"):   loss = model(batch)
    with stage("backward"):  accelerator.backward(loss)
    with stage("optimizer"): optimizer.step()

Emits an OpenTelemetry span when `opentelemetry-api` is importable and a
tracer provider is configured; otherwise records wall time per stage in
`STAGE_SECONDS`, which `train.py` folds into the metrics it logs. Stage-level
is the whole trace: operator- and kernel-level detail belongs to a profiler
switched on for a bounded window when a stage is anomalous.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager

STAGE_SECONDS: dict[str, float] = defaultdict(float)

try:
    from opentelemetry import trace as _otel_trace

    _tracer = _otel_trace.get_tracer("train")
except ImportError:  # the seam works without the dependency
    _tracer = None


@contextmanager
def stage(name: str) -> Iterator[None]:
    started = time.perf_counter()
    if _tracer is not None:
        with _tracer.start_as_current_span(name):
            yield
    else:
        yield
    STAGE_SECONDS[name] += time.perf_counter() - started


def drain_stage_seconds() -> dict[str, float]:
    """Return the accumulated per-stage seconds and reset them (call from the logging seam)."""
    snapshot = {f"time/{k}_s": v for k, v in STAGE_SECONDS.items()}
    STAGE_SECONDS.clear()
    return snapshot
