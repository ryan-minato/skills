"""Train <model> on <dataset>: one hand-written Accelerate loop."""

from pathlib import Path

import hydra
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import DictConfig
from torch.utils.data import DataLoader


def build_dataloader(cfg: DictConfig) -> DataLoader:
    dataset = <build the dataset from cfg.data>
    # The project's one sanctioned try/except lives in the dataset or
    # collate path: skip KNOWN-dirty samples, counting and logging every
    # skip. Everything else crashes.
    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )


def log_metrics(accelerator: Accelerator, step: int, metrics: dict) -> None:
    # The single logging seam. When a tracker is configured (pass
    # log_with= to Accelerator), switch to accelerator.log here — nothing
    # else in the loop changes.
    rendered = " ".join(f"{key}={value:.4g}" for key, value in metrics.items())
    accelerator.print(f"step {step}: {rendered}")


def evaluate(accelerator: Accelerator, model, dataloader) -> dict:
    # Plain function called from the loop — no callback framework.
    <compute and return metrics>


# config_path resolves relative to this file; run outputs are pinned to
# outputs/ in the Hydra config, not here.
@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)

    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.grad_accum_steps,
    )

    model = <build the model from cfg.model>
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optim.learning_rate)
    train_loader = build_dataloader(cfg)
    scheduler = <lr scheduler from cfg.optim, or None>

    # prepare() is the entire device story: no manual .to(device) anywhere.
    # Multi-GPU comes from launching with `accelerate launch train.py`.
    # CPU/disk offload and FSDP are turned on through `accelerate config`
    # or Accelerator kwargs — never by editing this loop; read the
    # big-model section of the Accelerate docs before reaching for them.
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    step = 0
    if cfg.resume_from:
        accelerator.load_state(cfg.resume_from)
        step = <restore the step counter, e.g. from the checkpoint dir name>

    model.train()
    for _epoch in range(cfg.epochs):
        for batch in train_loader:
            # No try/except around the step: a crash points at the bug,
            # wrapping it would only blur the traceback.
            with accelerator.accumulate(model):
                loss = <forward pass returning a scalar loss>
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

            step += 1
            if step % cfg.log_every_steps == 0:
                log_metrics(accelerator, step, {"loss": loss.item()})
            if step % cfg.checkpoint_every_steps == 0:
                accelerator.save_state(output_dir / "checkpoints" / f"step_{step}")

        log_metrics(accelerator, step, evaluate(accelerator, model, <eval loader>))

    # Final artifact: the unwrapped weights, loadable without Accelerate.
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    accelerator.save(unwrapped.state_dict(), output_dir / "model.pt")


if __name__ == "__main__":
    main()
