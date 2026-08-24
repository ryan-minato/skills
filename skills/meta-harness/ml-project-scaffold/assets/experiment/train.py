"""Train <model> on <dataset>: one hand-written Accelerate loop."""

from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from pydantic_settings import BaseSettings
from torch.utils.data import DataLoader


class Settings(BaseSettings):
    """The experiment's knobs — expose only values a run may change."""

    # Load values from config.yaml via a YAML settings source; wire it
    # per the current Pydantic Settings docs.
    seed: int = 42
    epochs: int = <n>
    batch_size: int = <n>
    learning_rate: float = <lr>
    grad_accum_steps: int = 1
    mixed_precision: str = "bf16"  # "fp16" or "no" where bf16 is unsupported
    num_workers: int = 4
    log_every_steps: int = 50
    checkpoint_every_steps: int = <n>
    resume_from: str | None = None
    output_dir: Path = Path("outputs")


def build_dataloader(settings: Settings) -> DataLoader:
    dataset = <build the dataset>
    # The project's one sanctioned try/except lives in the dataset or
    # collate path: skip KNOWN-dirty samples, counting and logging every
    # skip. Everything else crashes.
    return DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=settings.num_workers,
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


def main() -> None:
    settings = Settings()
    set_seed(settings.seed)

    accelerator = Accelerator(
        mixed_precision=settings.mixed_precision,
        gradient_accumulation_steps=settings.grad_accum_steps,
    )

    model = <build the model>
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)
    train_loader = build_dataloader(settings)
    scheduler = <lr scheduler, or None>

    # prepare() is the entire device story: no manual .to(device) anywhere.
    # Multi-GPU comes from launching with `accelerate launch train.py`.
    # CPU/disk offload and FSDP are turned on through `accelerate config`
    # or Accelerator kwargs — never by editing this loop; read the
    # big-model section of the Accelerate docs before reaching for them.
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    step = 0
    if settings.resume_from:
        accelerator.load_state(settings.resume_from)
        step = <restore the step counter, e.g. from the checkpoint dir name>

    model.train()
    for _epoch in range(settings.epochs):
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
            if step % settings.log_every_steps == 0:
                log_metrics(accelerator, step, {"loss": loss.item()})
            if step % settings.checkpoint_every_steps == 0:
                accelerator.save_state(
                    settings.output_dir / "checkpoints" / f"step_{step}"
                )

        log_metrics(accelerator, step, evaluate(accelerator, model, <eval loader>))

    # Final artifact: the unwrapped weights, loadable without Accelerate.
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    accelerator.save(unwrapped.state_dict(), settings.output_dir / "model.pt")


if __name__ == "__main__":
    main()
