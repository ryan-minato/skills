"""The configuration surface: every value a run may choose, as typed dataclasses.

`configs/*.yaml` hold named states; the command line overrides single values
(`optim.lr=1e-4`); `train.py` merges schema, file, and overrides into one
resolved document saved with the run. Expose only what experiments vary —
named choices, never import paths or control flow. Keep true constants in
code.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    train: str = "<immutable identity, e.g. hf:org/dataset@<revision> or data/raw/<name>@<checksum>>"
    eval: str = "<the evaluation set's identity — what 'better' is judged on>"
    batch_size: int = 32
    num_workers: int = 4


@dataclass
class ModelConfig:
    name: str = "<named choice the code supports, e.g. tiny | base>"
    pretrained: str | None = None  # an immutable identity, never a branch or `latest`


@dataclass
class OptimConfig:
    name: str = "adamw"  # adamw | sgd — the choices train.py builds; construction stays in code
    lr: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 0


@dataclass
class RunConfig:
    seed: int = 42
    steps: int = 1000
    grad_accum_steps: int = 1
    mixed_precision: str = "bf16"  # "fp16" or "no" where bf16 is unsupported
    log_every_steps: int = 50
    eval_every_steps: int = 500
    checkpoint_every_steps: int = 500
    resume_from: str | None = None  # a checkpoint directory; records the parent run
    output_dir: str = "outputs"
    tracker: str | None = None  # the tracker Accelerator logs to (log_with); None prints only


@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    run: RunConfig = field(default_factory=RunConfig)
