#!/usr/bin/env -S uv run --script
#
# /// script
# dependencies = [
#   "datasets",
#   "rich",
#   "huggingface_hub",
# ]
# ///

import argparse
import json

from datasets import load_dataset
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.tree import Tree

console = Console()


def inspect_dataset(dataset_name, config_name=None, split="train", num_rows=2):
    console.print(Panel(f"Streaming [bold cyan]{dataset_name}[/] ({split})", title="Dataset Inspector"))

    try:
        ds = load_dataset(dataset_name, config_name, split=split, streaming=True)

        # Get Features (Columns)
        iterator = iter(ds)
        first_example = next(iterator)

        # Display Structure
        tree = Tree("📁 Dataset Structure")
        cols_branch = tree.add("Columns (Features)")
        for key, value in first_example.items():
            type_str = type(value).__name__
            cols_branch.add(f"[bold green]{key}[/]: [italic]{type_str}[/]")
        console.print(tree)
        console.print("")

        # Display Rows
        console.print(f"[bold]First {num_rows} Examples:[/]")

        # Show first example (already fetched)
        json_str = json.dumps(first_example, indent=2, default=str)
        console.print(Syntax(json_str, "json", theme="monokai", line_numbers=True))

        # Show remaining requested rows
        for i in range(num_rows - 1):
            try:
                example = next(iterator)
                console.print(f"\n[dim]--- Row {i + 2} ---[/]")
                json_str = json.dumps(example, indent=2, default=str)
                console.print(Syntax(json_str, "json", theme="monokai", line_numbers=True))
            except StopIteration:
                break

    except Exception as e:
        console.print(f"[bold red]Error:[/]: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream Hugging Face datasets")
    parser.add_argument("name", type=str, help="Dataset path (e.g. 'HuggingFaceH4/ultrachat_200k')")
    parser.add_argument("--config", type=str, default=None, help="Dataset configuration")
    parser.add_argument("--split", type=str, default="train", help="Split to load")
    parser.add_argument("--rows", type=int, default=1, help="Rows to preview")

    args = parser.parse_args()
    inspect_dataset(args.name, args.config, args.split, args.rows)
