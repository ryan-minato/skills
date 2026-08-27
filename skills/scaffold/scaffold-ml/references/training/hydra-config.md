# Hydra Configuration

Read when creating or restructuring the `configs/` tree. Verify current
syntax and defaults against the Hydra docs, <https://hydra.cc/>, before
committing — the behaviors called out below have changed across major
versions.

## Minimal tree

```text
configs/
  config.yaml          # root: defaults list + top-level knobs
  model/<name>.yaml    # one file per model variant
  data/<name>.yaml     # one file per dataset/source
  optim/<name>.yaml    # optimizer/schedule variants
```

Start with only the groups the project varies today; a group with one
file is fine — it marks where variation is expected.

## Two path rules that bite

- **`config_path` is resolved relative to the file that declares
  `@hydra.main`, not the working directory.** A root-level `train.py`
  with `config_path="configs"` finds the tree no matter where the
  command is launched from. Confirm this against the current docs — it
  is the load-bearing fact of the wiring.
- **The run's output directory (and historically the working directory)
  is relocated per run by Hydra's defaults.** Pin it deliberately in
  `config.yaml` so artifacts land under `outputs/`, and resolve data
  paths from the project root in code — never from the process cwd.

## Exposure policy

Expose anticipatorily what planned experiments will vary — learning rate,
batch size, model and data selection, checkpoint cadence, paths — and
keep true constants in code. Anticipatory is still bounded: expose
values, never class paths to instantiate; a config that assembles modules
has stopped being a config.

## Worth knowing, not scaffolding

Multirun/sweeps and structured (dataclass-validated) configs exist when
the project grows into them; fetch their current state from the docs
rather than pre-wiring either.
