# The Configuration Surface

Read when creating or restructuring the configuration system, or when a
configuration file starts naming classes, registries, or conditionals.

## What configuration is

The source snapshot defines the program and the **configuration
surface** — the choices it allows a run to make. A run configuration
picks one state on that surface. Changing a value the code already
supports is a new run; adding a choice the code does not support changes
the source first.

## Default shape

Typed schema + YAML + command-line overrides + a resolved dump:

```python
from dataclasses import dataclass, field
from omegaconf import OmegaConf

@dataclass
class OptimConfig:
    name: str = "adamw"          # adamw | sgd — named choices the code supports
    lr: float = 3e-4
    weight_decay: float = 0.01

@dataclass
class TrainConfig:
    seed: int = 0
    steps: int = 10_000
    optim: OptimConfig = field(default_factory=OptimConfig)
    data: str = "hf:org/dataset@<revision>"   # an identity, not a branch

schema = OmegaConf.structured(TrainConfig)
cfg = OmegaConf.merge(schema, OmegaConf.load("configs/base.yaml"), OmegaConf.from_cli())
OmegaConf.save(OmegaConf.to_container(cfg, resolve=True), run_dir / "config.resolved.yaml")
```

- The schema is the legal surface: a key the schema lacks is an error,
  a wrong type is an error.
- YAML files hold named states (`configs/base.yaml`, `configs/small.yaml`);
  the command line overrides single values (`optim.lr=1e-4`).
- The resolved dump is saved with the run before training starts and is
  the run's configuration record. Verify the library's current merge,
  interpolation, and structured-config behavior from its documentation
  before relying on a specific rule.
- Alternatives that meet the same goal are fine when the project prefers
  them: dataclass plus argparse, YAML plus a validation library, a
  composition framework, plain Python. The goal is the invariant: a typed
  surface, named states, single-value overrides, one resolved document.

## What configuration is not

- **Not object construction**: `optimizer: {_target_: torch.optim.AdamW}`
  moves code into YAML. Name the choice (`optim.name: adamw`) and build
  the object in code from that name.
- **Not a registry**: a config that resolves strings to classes through a
  lookup table the reader must trace has become a second language.
- **Not control flow**: no conditionals, loops, or computed branches in
  configuration files; a value may interpolate another value, and that
  is the limit.
- **Not deep inheritance**: a base file and a few overrides are fine; a
  graph of files that must be mentally merged is not.

## Exposure

Expose what planned experiments will vary and what the environment
forces to vary (paths, device counts). Keep true constants in code. A
value that nobody varies is not configuration even when it belongs to
"the model".

## Existing composition framework

A project that already runs a composition framework keeps it. Two rules
limit it: no object instantiation from configuration (name choices,
build in code), and the per-run output directory is pinned so artifacts
land where the project expects; data paths resolve from the project
root, never from the process working directory.

## Vocabulary

"Hyperparameter" names a value that is one in the usual sense — learning
rate, weight decay, dropout, betas, warm-up. Dataset choice, optimizer
family, architecture, seed, and device count are data, algorithm, model,
randomness, and resource configuration; when searched, they are search
variables.
