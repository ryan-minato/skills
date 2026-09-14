"""Evaluate a run's model on the recorded evaluation set.

    just eval outputs/<run_id>        # writes outputs/<run_id>/eval/<eval_id>/{manifest.json,eval.json}

Bound to the benchmark AGENTS.md names: the evaluation set identity, the
metric definitions, and the evaluation code's commit are what make a
number comparable across runs. Uses the same `evaluate` function the
training loop calls, and records itself as a child run of the training
run (its own manifest, `parent_run_id` = the training run), because the
evaluation code's snapshot is part of the number's provenance.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

from accelerate import Accelerator
from omegaconf import OmegaConf

from run_manifest import finish_manifest, start_manifest
from train import evaluate, refuse_dirty_tree


def main() -> None:
    run_dir = Path(sys.argv[1])
    cfg = OmegaConf.load(run_dir / "config.resolved.yaml")  # the run's own configuration, never a fresh merge
    refuse_dirty_tree(cfg)
    accelerator = Accelerator()
    eval_id = f"{run_dir.name}-eval-{uuid.uuid4().hex[:6]}"
    eval_dir = run_dir / "eval" / eval_id
    manifest_path = None
    if accelerator.is_main_process:
        manifest_path = start_manifest(
            output_dir=eval_dir,
            run_id=eval_id,
            resolved_config_path=run_dir / "config.resolved.yaml",
            seed=cfg.run.seed,
            inputs=[
                {"name": "eval", "kind": "dataset", "identity": cfg.data.eval},
                {"name": "model", "kind": "checkpoint", "identity": str(run_dir / "model.pt")},
            ],
            parent_run_id=run_dir.name,
        )
    status = "failed"
    try:
        model = <build the model from cfg.model and load run_dir / "model.pt">
        eval_loader = <build the evaluation loader from cfg.data.eval — the identity recorded in the manifest>
        model, eval_loader = accelerator.prepare(model, eval_loader)
        metrics = evaluate(accelerator, model, eval_loader)
        if accelerator.is_main_process:
            (eval_dir / "eval.json").write_text(
                json.dumps({"run_id": run_dir.name, "eval": cfg.data.eval, **metrics}, indent=2)
            )
            accelerator.print(json.dumps(metrics))
        status = "complete"
    finally:
        if manifest_path is not None:
            finish_manifest(manifest_path, status=status, extra={"metrics": metrics} if status == "complete" else None)


if __name__ == "__main__":
    main()
