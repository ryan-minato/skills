# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "accelerate>=1.1.0",
#   "torch>=2.3.0",
# ]
# ///

import argparse

import torch
from accelerate import Accelerator


def make_dataloader(batch_size: int, feature_size: int, steps: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(steps):
        x = torch.randn(batch_size, feature_size)
        y = torch.randn(batch_size, 1)
        batches.append((x, y))
    return batches


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal gradient accumulation example")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=16)
    args = parser.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    model = torch.nn.Linear(32, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.8, total_iters=args.steps)
    dataloader = make_dataloader(args.batch_size, 32, args.steps)

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    for step, (x, y) in enumerate(dataloader, start=1):
        x = x.to(accelerator.device)
        y = y.to(accelerator.device)
        with accelerator.accumulate(model):
            pred = model(x)
            loss = torch.nn.functional.mse_loss(pred, y)
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if accelerator.is_main_process and step % args.gradient_accumulation_steps == 0:
            print(f"update_step={step // args.gradient_accumulation_steps}, loss={loss.item():.6f}")


if __name__ == "__main__":
    main()
