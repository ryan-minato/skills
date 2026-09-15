"""Train <model> on <dataset>: one explicit Accelerate loop.

    just train                      # configs/config.yaml
    just train optim.lr=1e-4        # one override
    accelerate launch train.py ...  # multi-device; the loop does not change

Every run leaves config.resolved.yaml, manifest.json, and manifest.running.json
under outputs/<run_id>/; metrics leave through log_metrics, stage time through
stage(). The <placeholders> are the project's; the seams around them are not.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, set_seed
from omegaconf import DictConfig, OmegaConf

from config import TrainConfig
from run_manifest import finish_manifest, source_snapshot, start_manifest
from stages import drain_stage_seconds, stage


def load_config(argv: list[str]) -> DictConfig:
    """Schema + named state + command-line overrides → one resolved configuration."""
    config_path = Path("configs/config.yaml")
    schema = OmegaConf.structured(TrainConfig)
    cfg = OmegaConf.merge(schema, OmegaConf.load(config_path), OmegaConf.from_cli(argv))
    # `run.mixed_precision=no` on the command line parses as YAML false; map it
    # back before the resolved dump so the record shows what ran.
    if str(cfg.run.mixed_precision).lower() in ("no", "false", "none"):
        cfg.run.mixed_precision = "no"
    return cfg  # a DictConfig validated against the schema


def refuse_dirty_tree(allow_dirty: bool) -> None:
    """A run from uncommitted code records a commit that is not the code that ran."""
    snapshot = source_snapshot()
    if snapshot["dirty"] and not allow_dirty:
        sys.exit(
            "train.py: the working tree has uncommitted changes. Commit the snapshot first "
            "(a throwaway run may pass run.allow_dirty=true; its manifest is marked degraded)."
        )


def manifest_params(manifest_path: Path) -> dict:
    """The manifest's identity scalars, in the shape a tracker takes as parameters."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "run_id": manifest["run_id"],
        "commit": manifest["source"]["commit"],
        "dirty": manifest["source"]["dirty"],
        "config_sha256": manifest["config"]["sha256"],
        "image_digest": manifest["environment"]["image_digest"],
        "lock_sha256": manifest["environment"]["lock_sha256"],
        "seed": manifest["randomness"]["seed"],
        "parent_run_id": manifest["parent_run_id"],
        "degraded": ",".join(manifest["degraded"]),
    }


def log_metrics(accelerator: Accelerator, step: int, metrics: dict) -> None:
    # The single logging seam: the tracker (Accelerator(log_with=...)) and the
    # console both hang off this function; nothing else in the loop logs.
    metrics = {**metrics, **drain_stage_seconds()}
    if accelerator.trackers:
        accelerator.log(metrics, step=step)
    rendered = " ".join(f"{key}={value:.4g}" for key, value in metrics.items())
    accelerator.print(f"step {step}: {rendered}")


def evaluate(accelerator: Accelerator, model, dataloader) -> dict:
    # Plain function called from the loop; the same code eval.py runs, so a
    # number in the loop and a number from `just eval` mean the same thing.
    <compute and return the benchmark metrics named in AGENTS.md>


def main() -> None:
    cfg = load_config(sys.argv[1:])
    refuse_dirty_tree(cfg.run.allow_dirty)  # every process sees the same tree, so every process exits together
    accelerator = Accelerator(
        mixed_precision=cfg.run.mixed_precision,
        gradient_accumulation_steps=cfg.run.grad_accum_steps,
        log_with=cfg.run.tracker,
    )
    # One run id per run, not per process: minted on the main process and
    # broadcast, so `accelerate launch` does not scatter one run over N dirs.
    run_id = broadcast_object_list([uuid.uuid4().hex[:12] if accelerator.is_main_process else None])[0]
    run_dir = Path(cfg.run.output_dir) / run_id
    resolved_path = run_dir / "config.resolved.yaml"

    manifest_path = None
    if accelerator.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, resolved_path, resolve=True)  # the run's configuration record
        manifest_path = start_manifest(
            output_dir=run_dir,
            run_id=run_id,
            resolved_config_path=resolved_path,
            seed=cfg.run.seed,
            inputs=[
                {"name": "train", "kind": "dataset", "identity": cfg.data.train},
                {"name": "eval", "kind": "dataset", "identity": cfg.data.eval},
                *(
                    [{"name": "pretrained", "kind": "model", "identity": cfg.model.pretrained}]
                    if cfg.model.pretrained
                    else []
                ),
            ],
            parent_run_id=<the run id of cfg.run.resume_from, or None>,
        )
    accelerator.wait_for_everyone()

    status = "failed"
    try:  # from here on, every exit finalizes the manifest
        if cfg.run.tracker:
            # The manifest's identity fields ride with the configuration as the
            # tracker's parameters, so a tracker run resolves to its manifest.
            params = {
                **OmegaConf.to_container(cfg, resolve=True),
                **(manifest_params(manifest_path) if manifest_path else {}),
            }
            accelerator.init_trackers("<project name>", config=params)

        set_seed(cfg.run.seed)
        model = <build the model from cfg.model — a named choice>
        optimizer = <build the optimizer from cfg.optim.name — construction stays in code>
        train_loader = <build the training DataLoader from cfg.data.train — the identity, not a branch>
        scheduler = <lr scheduler stepping after optimizer.step(), or None>

        # prepare() is the entire device story: no manual .to(device) anywhere.
        # Multi-GPU comes from `accelerate launch train.py`; offload and FSDP
        # come from `accelerate config` or Accelerator kwargs — never by
        # editing this loop.
        model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)

        step = 0  # optimizer steps, not micro-batches
        if cfg.run.resume_from:
            accelerator.load_state(cfg.run.resume_from)
            step = <restore the step counter, e.g. from the checkpoint dir name>

        grad_norm = None  # the clipping seam's return; None where a plugin clips for us
        model.train()
        iterator = iter(train_loader)
        while step < cfg.run.steps:
            with stage("data"):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(train_loader)
                    batch = next(iterator)
            with accelerator.accumulate(model):
                with stage("forward"):
                    loss = <forward pass returning a scalar loss>
                with stage("backward"):
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(model.parameters(), cfg.optim.max_grad_norm)
                with stage("optimizer"):
                    optimizer.step()  # a no-op on accumulation micro-batches
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()
            if not accelerator.sync_gradients:
                continue  # an accumulation micro-batch: no optimizer step happened
            step += 1
            if step % cfg.run.log_every_steps == 0:
                metrics = {"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]}
                if grad_norm is not None:
                    metrics["grad_norm"] = float(grad_norm)
                log_metrics(accelerator, step, metrics)
            if step % cfg.run.eval_every_steps == 0:
                log_metrics(accelerator, step, evaluate(accelerator, model, <eval loader>))
                model.train()  # evaluate() switches to eval mode; the next step must train again
            if step % cfg.run.checkpoint_every_steps == 0:
                accelerator.save_state(run_dir / "checkpoints" / f"step_{step}")

        # Final artifact: the unwrapped weights, loadable without Accelerate.
        accelerator.wait_for_everyone()
        accelerator.save(accelerator.unwrap_model(model).state_dict(), run_dir / "model.pt")
        status = "complete"
    finally:
        if manifest_path is not None:
            finish_manifest(manifest_path, status=status)
        if accelerator.trackers:
            accelerator.end_training()


if __name__ == "__main__":
    main()
