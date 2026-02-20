# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "accelerate>=1.1.0",
#   "torch>=2.3.0",
#   "torchvision>=0.18.0",
# ]
# ///

import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.datasets import FakeData
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, Resize, ToTensor

from accelerate import Accelerator


def parse_checkpointing_steps(value: str | None) -> str | int | None:
    if value is None:
        return None
    if value == "epoch":
        return value
    if value.isdigit():
        return int(value)
    raise ValueError(f"checkpointing_steps must be an integer or 'epoch'. Got: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete CV example inspired by accelerate/examples/complete_cv_example.py")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16", "fp8"])
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--with_tracking", action="store_true")
    parser.add_argument("--project_dir", type=str, default="logs")
    parser.add_argument("--output_dir", type=str, default="outputs/cv")
    parser.add_argument("--checkpointing_steps", type=str, default="epoch")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_train_batches", type=int, default=20)
    parser.add_argument("--max_eval_batches", type=int, default=10)
    args = parser.parse_args()

    checkpointing_steps = parse_checkpointing_steps(args.checkpointing_steps)
    accelerator = (
        Accelerator(cpu=args.cpu, mixed_precision=args.mixed_precision, log_with="all", project_dir=args.project_dir)
        if args.with_tracking
        else Accelerator(cpu=args.cpu, mixed_precision=args.mixed_precision)
    )
    if args.with_tracking:
        accelerator.init_trackers("complete_cv_classification_example", vars(args))

    os.makedirs(args.output_dir, exist_ok=True)

    train_transform = Compose(
        [Resize((224, 224)), RandomHorizontalFlip(), ToTensor(), Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )
    eval_transform = Compose([Resize((224, 224)), ToTensor(), Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    train_ds = FakeData(size=1024, image_size=(3, 224, 224), num_classes=10, transform=train_transform)
    eval_ds = FakeData(size=256, image_size=(3, 224, 224), num_classes=10, transform=eval_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)

    model = models.resnet18(weights=None, num_classes=10).to(accelerator.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        epochs=args.num_epochs,
        steps_per_epoch=min(len(train_loader), args.max_train_batches),
    )

    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )

    overall_step = 0
    if args.resume_from_checkpoint:
        accelerator.print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = torch.tensor(0.0, device=accelerator.device)
        for step, (images, labels) in enumerate(train_loader):
            if step >= args.max_train_batches:
                break
            images = images.to(accelerator.device)
            labels = labels.to(accelerator.device)
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            total_loss += loss.detach()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            overall_step += 1

            if isinstance(checkpointing_steps, int) and overall_step % checkpointing_steps == 0:
                accelerator.save_state(os.path.join(args.output_dir, f"step_{overall_step}"))

        model.eval()
        accurate = 0
        num_elems = 0
        for eval_step, (images, labels) in enumerate(eval_loader):
            if eval_step >= args.max_eval_batches:
                break
            images = images.to(accelerator.device)
            labels = labels.to(accelerator.device)
            with torch.no_grad():
                logits = model(images)
            predictions = logits.argmax(dim=-1)
            predictions, references = accelerator.gather_for_metrics((predictions, labels))
            accurate += (predictions == references).long().sum().item()
            num_elems += references.shape[0]

        accuracy = accurate / max(num_elems, 1)
        accelerator.print(f"epoch={epoch} accuracy={accuracy:.4f}")
        if args.with_tracking:
            accelerator.log(
                {"epoch": epoch, "accuracy": accuracy, "train_loss": (total_loss / max(args.max_train_batches, 1)).item()},
                step=overall_step,
            )
        if checkpointing_steps == "epoch":
            accelerator.save_state(os.path.join(args.output_dir, f"epoch_{epoch}"))

    accelerator.end_training()


if __name__ == "__main__":
    main()
