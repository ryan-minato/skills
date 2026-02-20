# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "accelerate>=1.1.0",
#   "datasets>=3.1.0",
#   "evaluate>=0.4.0",
#   "torch>=2.3.0",
#   "transformers>=4.50.0",
# ]
# ///

import argparse
import os

import evaluate
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed

from accelerate import Accelerator, DistributedType


MAX_GPU_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32


def parse_checkpointing_steps(value: str | None) -> str | int | None:
    if value is None:
        return None
    if value == "epoch":
        return value
    if value.isdigit():
        return int(value)
    raise ValueError(f"checkpointing_steps must be an integer or 'epoch'. Got: {value}")


def build_dataloaders(accelerator: Accelerator, tokenizer: AutoTokenizer, batch_size: int) -> tuple[DataLoader, DataLoader]:
    datasets = load_dataset("glue", "mrpc")
    datasets["train"] = datasets["train"].select(range(min(len(datasets["train"]), 256)))
    datasets["validation"] = datasets["validation"].select(range(min(len(datasets["validation"]), 128)))

    def tokenize_function(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

    with accelerator.main_process_first():
        tokenized = datasets.map(tokenize_function, batched=True, remove_columns=["idx", "sentence1", "sentence2"])
    tokenized = tokenized.rename_column("label", "labels")

    def collate(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    train_loader = DataLoader(tokenized["train"], shuffle=True, collate_fn=collate, batch_size=batch_size)
    eval_loader = DataLoader(tokenized["validation"], shuffle=False, collate_fn=collate, batch_size=EVAL_BATCH_SIZE)
    return train_loader, eval_loader


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete NLP example inspired by accelerate/examples/complete_nlp_example.py")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16", "fp8"])
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--with_tracking", action="store_true")
    parser.add_argument("--project_dir", type=str, default="logs")
    parser.add_argument("--output_dir", type=str, default="outputs/nlp")
    parser.add_argument("--checkpointing_steps", type=str, default="epoch")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
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
        accelerator.init_trackers("complete_nlp_mrpc_example", vars(args))

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    metric = evaluate.load("glue", "mrpc")

    batch_size = args.batch_size
    gradient_accumulation_steps = 1
    if batch_size > MAX_GPU_BATCH_SIZE and accelerator.distributed_type != DistributedType.XLA:
        gradient_accumulation_steps = batch_size // MAX_GPU_BATCH_SIZE
        batch_size = MAX_GPU_BATCH_SIZE

    train_loader, eval_loader = build_dataloaders(accelerator, tokenizer, batch_size)
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", return_dict=True).to(accelerator.device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=10,
        num_training_steps=(min(len(train_loader), args.max_train_batches) * args.num_epochs) // gradient_accumulation_steps,
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
        for step, batch in enumerate(train_loader):
            if step >= args.max_train_batches:
                break
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / gradient_accumulation_steps
            total_loss += loss.detach()
            accelerator.backward(loss)
            if step % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            overall_step += 1

            if isinstance(checkpointing_steps, int) and overall_step % checkpointing_steps == 0:
                accelerator.save_state(os.path.join(args.output_dir, f"step_{overall_step}"))

        model.eval()
        for eval_step, batch in enumerate(eval_loader):
            if eval_step >= args.max_eval_batches:
                break
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
            metric.add_batch(predictions=predictions, references=references)

        eval_metric = metric.compute()
        accelerator.print(f"epoch={epoch} eval={eval_metric}")
        if args.with_tracking:
            accelerator.log(
                {
                    "epoch": epoch,
                    "accuracy": eval_metric["accuracy"],
                    "f1": eval_metric["f1"],
                    "train_loss": (total_loss / max(args.max_train_batches, 1)).item(),
                },
                step=overall_step,
            )
        if checkpointing_steps == "epoch":
            accelerator.save_state(os.path.join(args.output_dir, f"epoch_{epoch}"))

    accelerator.end_training()


if __name__ == "__main__":
    main()
