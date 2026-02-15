#!/usr/bin/env -S uv run --script
#
# /// script
# dependencies = [
#   "rich",
# ]
# ///

import argparse

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def estimate_vram(model_size_b, precision="bf16", context_len=4096, batch_size=1, train_type="full", optimizer="adamw"):
    # Constants
    BYTES_PER_PARAM = {"fp32": 4, "bf16": 2, "fp16": 2, "int8": 1, "int4": 0.5}
    bpp = BYTES_PER_PARAM.get(precision, 2)

    # 1. Model Weights
    model_mem = model_size_b * bpp

    # 2. KV Cache (Inference / Generation)
    # Approx: 2 (k/v) * layers * hidden * seq * batch * bytes
    # Simplifying rule of thumb for standard Llama architecture:
    # ~0.5MB per token per layer for float16? Let's use a rough estimator:
    # 7B model has ~32 layers, 4096 hidden.
    # KV Cache per token approx = 2 * 32 * 4096 * 2 bytes = 524 KB / token
    # For N billion params, roughly scales.
    kv_cache_mem = context_len * batch_size * model_size_b * 0.08  # Rough heuristic in GB

    # 3. Training Overheads
    grad_mem = 0
    optim_mem = 0
    activation_mem = 0

    if train_type != "inference":
        # Gradients (always fp32 or fp16, usually equal to params)
        if train_type == "lora":
            # LoRA trainable params ~2%
            trainable_params = model_size_b * 0.02
            grad_mem = trainable_params * 2
            optim_mem = trainable_params * (8 if optimizer == "adamw" else 2)  # 8 bytes for AdamW states
        else:
            # Full fine-tuning
            grad_mem = model_size_b * 2  # Gradients
            if optimizer == "adamw":
                optim_mem = model_size_b * 8
            elif optimizer == "adamw_8bit":
                optim_mem = model_size_b * 2
            elif optimizer == "sgd":
                optim_mem = model_size_b * 4

        # Activations (Very rough estimate, heavily depends on seq_len & checkpointing)
        # Activation memory grows linearly with seq_len
        activation_mem = context_len * batch_size * model_size_b * 0.05
        if "gradient_checkpointing" in train_type:
            activation_mem /= 5  # Checkpointing saves massive memory

    total_mem = model_mem + kv_cache_mem + grad_mem + optim_mem + activation_mem

    # Visualization
    table = Table(title=f"VRAM Estimator: {model_size_b}B Model ({precision})", box=box.ROUNDED)
    table.add_column("Component", style="cyan")
    table.add_column("Memory (GB)", justify="right", style="green")

    table.add_row("Model Weights", f"{model_mem:.2f}")
    if train_type == "inference":
        table.add_row(f"KV Cache (ctx={context_len})", f"{kv_cache_mem:.2f}")
    else:
        table.add_row("Gradients", f"{grad_mem:.2f}")
        table.add_row(f"Optimizer ({optimizer})", f"{optim_mem:.2f}")
        table.add_row("Activations (Est.)", f"{activation_mem:.2f}")

    table.add_section()
    table.add_row("[bold]TOTAL Estimated[/]", f"[bold yellow]{total_mem:.2f} GB[/]")

    console.print(table)

    # Recommendations
    if total_mem > 80:
        console.print(
            Panel(
                "[bold red]Recommendation:[/]\nRequires [bold]A100/H100 (80GB)[/] or [bold]Multi-GPU (FSDP/ZeRO-3)[/]"
            )
        )
    elif total_mem > 24:
        console.print(
            Panel(
                "[bold yellow]Recommendation:[/]\nRequires [bold]A100 (40GB)[/] or [bold]A6000/6000 Ada[/].\nConsider [bold]Quantization (4-bit)[/] or [bold]ZeRO-Offload[/] for consumer GPUs."
            )
        )
    else:
        console.print(Panel("[bold green]Recommendation:[/]\nFits on consumer GPUs (RTX 3090/4090)."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=float, required=True, help="Model size in Billions")
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16", "fp16", "int8", "int4"])
    parser.add_argument("--mode", type=str, default="training", choices=["training", "inference", "lora"])
    parser.add_argument("--optim", type=str, default="adamw", choices=["adamw", "adamw_8bit", "sgd"])

    args = parser.parse_args()
    estimate_vram(args.size, args.precision, args.context, args.batch, args.mode, args.optim)
