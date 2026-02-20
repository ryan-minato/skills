# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "accelerate>=1.1.0",
#   "datasets>=3.1.0",
#   "torch>=2.3.0",
#   "transformers>=4.50.0",
# ]
# ///

import argparse
import math

import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, get_scheduler

from accelerate import Accelerator


def build_lm_dataloader(tokenizer: AutoTokenizer, dataset_name: str, dataset_config_name: str, block_size: int, batch_size: int):
    raw = load_dataset(dataset_name, dataset_config_name)
    train_split = raw["train"].select(range(min(len(raw["train"]), 512)))
    eval_split = raw["validation"].select(range(min(len(raw["validation"]), 256)))

    def tokenize(examples):
        return tokenizer(examples["text"])

    tokenized_train = train_split.map(tokenize, batched=True, remove_columns=train_split.column_names)
    tokenized_eval = eval_split.map(tokenize, batched=True, remove_columns=eval_split.column_names)

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = (len(concatenated["input_ids"]) // block_size) * block_size
        result = {
            k: [v[i : i + block_size] for i in range(0, total_length, block_size)] for k, v in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_train = tokenized_train.map(group_texts, batched=True)
    lm_eval = tokenized_eval.map(group_texts, batched=True)

    train_loader = DataLoader(lm_train, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size)
    eval_loader = DataLoader(lm_eval, shuffle=False, collate_fn=default_data_collator, batch_size=batch_size)
    return train_loader, eval_loader


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal LM no-trainer template inspired by run_clm_no_trainer.py")
    parser.add_argument("--model_name_or_path", type=str, default="gpt2")
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config_name", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=50)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16", "fp8"])
    args = parser.parse_args()

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader, eval_loader = build_lm_dataloader(
        tokenizer,
        args.dataset_name,
        args.dataset_config_name,
        args.block_size,
        args.batch_size,
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    steps_per_epoch = math.ceil(min(len(train_loader), args.max_train_steps) / args.gradient_accumulation_steps)
    total_steps = min(args.max_train_steps, steps_per_epoch * args.num_train_epochs)
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=max(total_steps // 10, 1),
        num_training_steps=total_steps,
    )

    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )

    completed_steps = 0
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            if completed_steps >= total_steps:
                break
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                completed_steps += 1
                if accelerator.is_main_process and completed_steps % 10 == 0:
                    print({"step": completed_steps, "train_loss": float(loss.detach().item())})

        model.eval()
        losses = []
        for eval_step, batch in enumerate(eval_loader):
            if eval_step >= 20:
                break
            with torch.no_grad():
                outputs = model(**batch)
            losses.append(accelerator.gather_for_metrics(outputs.loss.repeat(args.batch_size)))

        losses = torch.cat(losses)
        try:
            perplexity = torch.exp(losses.mean()).item()
        except OverflowError:
            perplexity = float("inf")
        accelerator.print({"epoch": epoch, "perplexity": perplexity})

    accelerator.end_training()


if __name__ == "__main__":
    main()
