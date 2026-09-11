# The Configuration Surface

Read when creating or restructuring `configs/`, when a configuration
file starts naming classes or conditionals, or when the project already
runs Hydra.

## What configuration is

The source snapshot defines the program and the configuration surface —
the choices a run may make. A run configuration picks one state on that
surface. Changing a value the code already supports is a new run; adding
a choice the code does not support changes the source first.

## The default shape

`config.py` holds the schema as dataclasses (the
legal surface: an unknown key or a wrong type is an error);
`configs/config.yaml` holds a named
state; the command line overrides single values (`optim.lr=1e-4`); the
entry point merges schema, file, and overrides and writes the resolved
document to `outputs/<run_id>/config.resolved.yaml` before the first
step, and the manifest records its hash.

```python
schema = OmegaConf.structured(TrainConfig)
cfg = OmegaConf.merge(schema, OmegaConf.load(config_path), OmegaConf.from_cli(overrides))
OmegaConf.save(cfg, run_dir / "config.resolved.yaml", resolve=True)
```

Verify the library's current merge, interpolation, and structured-config
behavior from its documentation before relying on a specific rule.

## Exposure

Expose what planned experiments will vary and what the environment
forces to vary (paths, device counts); keep true constants in code. A
configuration may select among named, code-supported choices
(`optim.name: adamw | sgd`). It may not construct arbitrary objects,
compose through registries or import paths, carry control flow, or
inherit deeply — that is a second programming language. "Hyperparameter"
names a value that is one in the usual sense; a dataset choice, an
optimizer family, an architecture variant, a seed, or a device count is
data, algorithm, model, randomness, or resource configuration, and when
searched they are search variables.

## Existing Hydra

A project that already runs Hydra keeps it. Two rules limit it: no object
instantiation from configuration (name choices, build in code), and the
per-run output directory is pinned so artifacts land under `outputs/`;
data paths resolve from the project root, never from the process working
directory. `config_path` resolves relative to the file that declares the
entry point — confirm against the current documentation. Multirun and
structured configs exist when the project grows into them; fetch their
current state rather than pre-wiring either.
