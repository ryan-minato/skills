# Choosing the Framework

Read before fixing the framework of a project that has not settled one,
and whenever the inventory shows a JAX signal (below). A project that
already trains on a framework keeps it: scaffolding is not permission to
migrate.

## The default

PyTorch with Accelerate and an explicit loop, for every ordinary model
training or fine-tuning project. Fix it without asking when no signal
below is present; record it in `AGENTS.md`.

## Signals that call for a JAX evaluation

Evaluate JAX against PyTorch when the inventory shows one or more of:

- TPU hardware, now or as the stated target.
- A differentiable simulation or solver: the whole computation, not only
  a model, must be differentiated.
- Scientific ML or numerical research where the computation matters more
  than the model — mathematical functions, numerical algorithms,
  physics-informed or scientific-computing workloads.
- Higher-order automatic differentiation as a routine operation.
- Heavy composition of `grad`, `vmap`, and `jit` — per-example gradients,
  batched Jacobians, function transformations as the program's shape.
- Large-scale homogeneous parallel computation (the same pure function
  over many devices or shards).
- Compiler or automatic-differentiation research itself.

None of these decides on its own; the evidence for each comes from the
inventory — hardware, existing code and dependencies, the research
objective, and the team's stated experience — never from the framework's
reputation.

## The evaluation and the decision

Write the evaluation in a few lines for the user: the signals found and
their evidence, what each framework costs here (the ecosystem the
project depends on — pretrained weights, data tooling, the team's
fluency — against the transformations and hardware the work needs), and
one recommendation with its reason. The user decides; record the chosen
framework and the deciding signal in `AGENTS.md` so no later builder
reopens it. A recommendation is never applied on its own.

## What changes when JAX is chosen

Everything framework-neutral is deposited unchanged: the configuration
surface, the manifest (its runtime facts already record `jax` and its
backend), the tracker selection, the research-task convention, the test
markers, Ruff, hooks, and the container recipe (a base image with the
matching CUDA or TPU runtime instead of the PyTorch-routed one; see the
JAX section of `references/hardware-deps.md`). The Accelerate loop asset
(`train.py`) is PyTorch-only; write the training entry point from the
shape below with the same seams, and keep `eval.py` bound to the
benchmark the same way.

The JAX loop shape — verify current APIs against the libraries' own
documentation at build time, never from memory:

- The training step is one pure function of `(params, opt_state, batch,
  key)` returning the updated `(params, opt_state)` and the metrics,
  compiled once with `jit`; state is explicit and immutable, never held
  by objects.
- Randomness is an explicit key, split per step and recorded with the
  seed; the manifest's seed is the root key.
- Parameters and optimizer state come from the project's chosen neural
  network and optimizer libraries (Flax and Optax are the mature
  first-party choices); checkpoints go through a checkpoint library
  (Orbax) under `outputs/<run_id>/checkpoints/`, with the step recorded
  so resume restores the counter.
- Multi-device runs use `jit` sharding or `pmap` decided in one place at
  the top of the entry point, never inside the step.
- The logging seam (`log_metrics`) and the stage-trace seam (`stages.py`)
  are the same; metrics are logged against the optimizer step after
  device-to-host transfer, never per micro-batch inside the compiled step.
- The data pipeline may stay on a PyTorch or Grain loader; only the step
  must be JAX. Record the choice and its reason.
