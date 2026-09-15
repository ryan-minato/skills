"""The configuration surface: every value a run may choose, as typed dataclasses.

Named states live in `configs/*.yaml`; the command line overrides single values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class DataConfig:
    train: str = "<immutable identity, e.g. hf:org/dataset@<revision> or data/raw/<name>@<checksum>>"
    eval: str = "<the evaluation set's identity — what 'better' is judged on>"


@dataclass
class ModelConfig:
    name: str = "<named choice the code supports>"
    pretrained: str | None = None  # an immutable identity, never a branch or `latest`


@dataclass
class OptimConfig:
    name: str = "<named choice train.py builds>"  # construction stays in code
    lr: float = MISSING
    max_grad_norm: float = MISSING  # the clipping seam's bound


@dataclass
class RunConfig:
    seed: int = 42
    steps: int = MISSING  # optimizer steps
    grad_accum_steps: int = 1
    mixed_precision: str = "no"  # bf16 | fp16 | no
    log_every_steps: int = 50
    eval_every_steps: int = 500
    checkpoint_every_steps: int = 500
    resume_from: str | None = None  # a checkpoint directory; records the parent run
    output_dir: str = "outputs"
    tracker: str | None = None  # the tracker Accelerator logs to (log_with); None prints only
    allow_dirty: bool = False  # a throwaway run from uncommitted code; the manifest marks it degraded


@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    run: RunConfig = field(default_factory=RunConfig)
