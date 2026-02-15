#!/usr/bin/env -S uv run --script
#
# /// script
# dependencies = [
#   "torch",
#   "transformers",
#   "accelerate",
#   "rich",
#   "psutil",
# ]
# ///

import platform
import sys

import pkg_resources
import psutil
import torch
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def get_lib_version(lib_name):
    try:
        return pkg_resources.get_distribution(lib_name).version
    except pkg_resources.DistributionNotFound:
        return "[red]Not Installed[/]"


def check_env():
    console.print(Panel.fit("[bold cyan]Hugging Face Environment Health Check[/]", box=box.ROUNDED))

    # 1. System Info
    sys_table = Table(title="System Information", show_header=False, box=box.SIMPLE)
    sys_table.add_row("OS", platform.platform())
    sys_table.add_row("Python", sys.version.split()[0])
    sys_table.add_row("RAM", f"{psutil.virtual_memory().total / (1024**3):.2f} GB")
    console.print(sys_table)

    # 2. Libraries
    lib_table = Table(title="Critical Libraries", box=box.SIMPLE)
    lib_table.add_column("Library", style="cyan")
    lib_table.add_column("Version", style="green")

    libs = [
        "torch",
        "transformers",
        "accelerate",
        "peft",
        "trl",
        "datasets",
        "bitsandbytes",
        "deepspeed",
        "flash_attn",
    ]

    for lib in libs:
        lib_table.add_row(lib, get_lib_version(lib))
    console.print(lib_table)

    # 3. GPU / CUDA
    console.print("\n[bold]GPU & CUDA Diagnostic[/]")
    if torch.cuda.is_available():
        gpu_table = Table(box=box.MINIMAL)
        gpu_table.add_column("ID")
        gpu_table.add_column("Name")
        gpu_table.add_column("VRAM (GB)")
        gpu_table.add_column("Arch")
        gpu_table.add_column("BF16 Support")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram = props.total_memory / (1024**3)
            bf16 = "[green]Yes[/]" if torch.cuda.is_bf16_supported() else "[red]No[/]"
            gpu_table.add_row(str(i), props.name, f"{vram:.1f}", f"{props.major}.{props.minor}", bf16)

        console.print(gpu_table)
        console.print(f"CUDA Version: [yellow]{torch.version.cuda}[/]")

        # P2P Check
        if torch.cuda.device_count() > 1:
            console.print("\n[bold]P2P Interconnect (NVLink/PCIe)[/]")
            p2p_table = Table(show_header=True, header_style="bold magenta")
            p2p_table.add_column("GPU")
            for i in range(torch.cuda.device_count()):
                p2p_table.add_column(f"GPU {i}")

            for i in range(torch.cuda.device_count()):
                row = [f"GPU {i}"]
                for j in range(torch.cuda.device_count()):
                    if i == j:
                        row.append("-")
                    else:
                        try:
                            can_access = torch.cuda.device.can_device_access_peer(i, j)
                            row.append("[green]YES[/]" if can_access else "[red]NO[/]")
                        except Exception:
                            row.append("[red]ERR[/]")
                p2p_table.add_row(*row)
            console.print(p2p_table)

    else:
        console.print("[bold red]NO CUDA DEVICES DETECTED[/]. Running on CPU (Training will be slow).")


if __name__ == "__main__":
    check_env()
