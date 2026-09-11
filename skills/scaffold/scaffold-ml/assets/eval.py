"""Evaluate a run's model on the recorded evaluation set.

    just eval outputs/<run_id>        # the run to evaluate; writes outputs/<run_id>/eval.json

Bound to the benchmark AGENTS.md names: the evaluation set identity, the
metric definitions, and the evaluation code's commit are what make a
number comparable across runs. Uses the same `evaluate` function the
training loop calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from accelerate import Accelerator
from omegaconf import OmegaConf

from train import evaluate


def main() -> None:
    run_dir = Path(sys.argv[1])
    cfg = OmegaConf.load(run_dir / "config.resolved.yaml")  # the run's own configuration, never a fresh merge
    accelerator = Accelerator()
    model = <build the model from cfg.model and load run_dir / "model.pt">
    eval_loader = <build the evaluation loader from cfg.data.eval — the identity recorded in the manifest>
    model, eval_loader = accelerator.prepare(model, eval_loader)
    metrics = evaluate(accelerator, model, eval_loader)
    if accelerator.is_main_process:
        (run_dir / "eval.json").write_text(json.dumps({"run_id": run_dir.name, "eval": cfg.data.eval, **metrics}, indent=2))
        accelerator.print(json.dumps(metrics))


if __name__ == "__main__":
    main()
