#!/usr/bin/env python3
"""
Bootstrap script to create the hf-llm-architect skill structure and files.
This script creates all necessary directories and files for the skill.

Usage:
    python3 create_hf_llm_architect.py
"""

import os
from pathlib import Path

# Base path
BASE_DIR = Path(__file__).parent / "skills" / "hf-llm-architect"

# Directory structure
DIRS = [
    "",
    "scripts",
    "references/01_pretraining",
    "references/02_fine_tuning",
    "references/03_alignment",
    "references/04_distributed",
    "references/05_inference",
]

# File contents dictionary
FILES = {
    "SKILL.md": """---
name: hf-llm-architect
description: An expert LLM development guide for the Hugging Face ecosystem. Features PEP 723 compliant tools for environment checking, data inspection, and VRAM estimation, alongside in-depth guides on Scaling Laws (Step/Farseer), GRPO/DPO, and distributed training.
compatibility: Requires python3, uv (recommended for script execution)
---

# Hugging Face LLM Architect

This skill provides a production-ready workflow for training and deploying LLMs using Hugging Face libraries (`transformers`, `trl`, `peft`, `accelerate`). It incorporates the latest research on scaling laws and reasoning alignment.

## 🛠️ Operational Tools (PEP 723 Compliant)

These scripts utilize `uv` for dependency management. You can run them directly if `uv` is installed.

### 1. Environment & Hardware Check
Diagnose your GPU setup, CUDA versions, and interconnects (P2P/NVLink).
```bash
# Run directly (dependencies auto-installed)
./scripts/env_health_check.py
```

### 2. Dataset Stream Inspection

Preview massive datasets from the Hub without downloading them. Checks column structures for Chat Templates.

```bash
# Example: Inspect the UltraChat dataset
./scripts/stream_data_preview.py HuggingFaceH4/ultrachat_200k --split train_sft
```

### 3. VRAM Requirement Calculator

Estimate memory usage for Training (Full/LoRA) and Inference based on model size and context length.

```bash
# Example: 8B model, 8192 context, fp16 precision
./scripts/calc_vram_requirements.py --size 8 --context 8192 --precision fp16 --mode training
```

---

## 📚 Technical Knowledge Base

### Phase 1: Pre-training & Scaling

* **Theory**: Understand **Step Law** (Hyperparameters) and **Farseer** (Loss Prediction) to optimize compute.
* **Practice**: Data mixing recipes for continual pre-training.
* 🔗 [Scaling Laws & Recipes](references/01_pretraining/scaling_laws_step_farseer.md)
* 🔗 [Continual Pre-training Guide](references/01_pretraining/continual_pretraining_recipes.md)

### Phase 2: Supervised Fine-Tuning (SFT)

* **Core**: Chat Templates, Sequence Packing (`packing=True`), and Loss Masking.
* **Efficiency**: LoRA/QLoRA configuration (`target_modules="all-linear"`).
* 🔗 [SFT Best Practices](references/02_fine_tuning/sft_best_practices.md)
* 🔗 [PEFT & LoRA Config](references/02_fine_tuning/peft_lora_config.md)

### Phase 3: Alignment & Reasoning

* **General**: DPO vs ORPO for chat preferences.
* **Reasoning**: **GRPO** (DeepSeek-R1 style) for math/code tasks without a value model.
* 🔗 [Preference Optimization (DPO/ORPO)](references/03_alignment/dpo_orpo_preference.md)
* 🔗 [Reasoning with GRPO](references/03_alignment/grpo_reasoning.md)

### Phase 4: Infrastructure

* **Distributed**: Choosing between DDP, DeepSpeed ZeRO-1/2/3, and FSDP.
* **Inference**: Flash Attention 2, KV Cache Quantization, and vLLM serving.
* 🔗 [Distributed Strategy Matrix](references/04_distributed/strategy_matrix.md)
* 🔗 [DeepSpeed & FSDP Config](references/04_distributed/deepspeed_fsdp_config.md)
* 🔗 [Inference Optimization](references/05_inference/production_optimization.md)
""",

    "scripts/env_health_check.py": '''#!/usr/bin/env -S uv run --script
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

import torch
import sys
import platform
import psutil
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
import pkg_resources

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
        "torch", "transformers", "accelerate", "peft", 
        "trl", "datasets", "bitsandbytes", "deepspeed", "flash_attn"
    ]
    
    for lib in libs:
        lib_table.add_row(lib, get_lib_version(lib))
    console.print(lib_table)

    # 3. GPU / CUDA
    console.print("\\n[bold]GPU & CUDA Diagnostic[/]")
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
            console.print("\\n[bold]P2P Interconnect (NVLink/PCIe)[/]")
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
                        except:
                            row.append("[red]ERR[/]")
                p2p_table.add_row(*row)
            console.print(p2p_table)

    else:
        console.print("[bold red]NO CUDA DEVICES DETECTED[/]. Running on CPU (Training will be slow).")

if __name__ == "__main__":
    check_env()
''',

    "scripts/stream_data_preview.py": '''#!/usr/bin/env -S uv run --script
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
from rich.tree import Tree
from rich.syntax import Syntax
from rich.panel import Panel

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
                console.print(f"\\n[dim]--- Row {i+2} ---[/]")
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
''',

    "scripts/calc_vram_requirements.py": '''#!/usr/bin/env -S uv run --script
#
# /// script
# dependencies = [
#   "rich",
# ]
# ///

import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

def estimate_vram(model_size_b, precision="bf16", context_len=4096, batch_size=1, train_type="full", optimizer="adamw"):
    
    # Constants
    BYTES_PER_PARAM = {
        "fp32": 4, "bf16": 2, "fp16": 2, "int8": 1, "int4": 0.5
    }
    bpp = BYTES_PER_PARAM.get(precision, 2)
    
    # 1. Model Weights
    model_mem = model_size_b * bpp
    
    # 2. KV Cache (Inference / Generation)
    # Approx: 2 (k/v) * layers * hidden * seq * batch * bytes
    # Simplifying rule of thumb for standard Llama architecture: 
    # 7B model has ~32 layers, 4096 hidden. 
    # KV Cache per token approx = 2 * 32 * 4096 * 2 bytes = 524 KB / token
    # For N billion params, roughly scales. 
    kv_cache_mem = (context_len * batch_size * model_size_b * 0.08) # Rough heuristic in GB
    
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
            optim_mem = trainable_params * (8 if optimizer == "adamw" else 2) # 8 bytes for AdamW states
        else:
            # Full fine-tuning
            grad_mem = model_size_b * 2 # Gradients
            if optimizer == "adamw":
                optim_mem = model_size_b * 8
            elif optimizer == "adamw_8bit":
                optim_mem = model_size_b * 2
            elif optimizer == "sgd":
                optim_mem = model_size_b * 4
        
        # Activations (Very rough estimate, heavily depends on seq_len & checkpointing)
        # Activation memory grows linearly with seq_len
        activation_mem = (context_len * batch_size * model_size_b * 0.05) 
        if "gradient_checkpointing" in train_type:
             activation_mem /= 5 # Checkpointing saves massive memory

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
        console.print(Panel("[bold red]Recommendation:[/]\\nRequires [bold]A100/H100 (80GB)[/] or [bold]Multi-GPU (FSDP/ZeRO-3)[/]"))
    elif total_mem > 24:
        console.print(Panel("[bold yellow]Recommendation:[/]\\nRequires [bold]A100 (40GB)[/] or [bold]A6000/6000 Ada[/].\\nConsider [bold]Quantization (4-bit)[/] or [bold]ZeRO-Offload[/] for consumer GPUs."))
    else:
         console.print(Panel("[bold green]Recommendation:[/]\\nFits on consumer GPUs (RTX 3090/4090)."))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=float, required=True, help="Model size in Billions")
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16", "int8", "int4"])
    parser.add_argument("--mode", type=str, default="training", choices=["training", "inference", "lora"])
    parser.add_argument("--optim", type=str, default="adamw", choices=["adamw", "adamw_8bit", "sgd"])
    
    args = parser.parse_args()
    estimate_vram(args.size, args.precision, args.context, args.batch, args.mode, args.optim)
''',

    "references/01_pretraining/scaling_laws_step_farseer.md": """# Scaling Laws: Step Law & Farseer

Optimizing compute and hyperparameters before training.

## 1. Step Law (Hyperparameter Optimization)
Step Law focuses on finding the optimal Learning Rate ($lr$) and Batch Size ($B$) for a given model size ($N$) and data size ($D$).

### Core Formulas
* **Learning Rate**: $lr_{opt} \\propto N^{-\\alpha} \\cdot D^{-\\beta}$
    * *Insight*: Larger models generally require **lower** learning rates.
* **Batch Size**: $B_{opt} \\propto D^{\\gamma}$
    * *Insight*: Batch size should grow as you train on more tokens.

### Practical Application
Do not guess the LR. Train a small proxy model (e.g., 100M parameters) on a subset of your data to find its optimal LR. Then, use Step Law coefficients to extrapolate the LR for your target 7B/70B model.

## 2. Farseer (Performance Prediction)
Farseer improves upon Chinchilla by modeling the Loss Surface $L(N, D)$.

* **Variable Compute**: Instead of training one model size to convergence, train multiple small model sizes for varying numbers of steps.
* **Surface Fitting**: Fit the Farseer equation to these points to predict the loss of a much larger model.
* **Efficiency**: Reduces the extrapolation error by ~4x compared to Chinchilla, potentially saving millions in wasted compute on models that won't converge as expected.
""",

    "references/01_pretraining/continual_pretraining_recipes.md": """# Continual Pre-training Recipes

How to inject new knowledge (e.g., a new language or domain) into an existing LLM without breaking it.

## 1. The Data Mixture
The biggest risk is **Catastrophic Forgetting**.
* **Target Data**: 90% - 95% (e.g., Technical Manuals, Legal Texts).
* **Replay Data**: 5% - 10% (General English/Code, e.g., FineWeb-Edu, Cosmopedia).
    * *Note*: Without replay data, the model's reasoning and basic grammar capabilities will degrade rapidly.

## 2. Tokenizer Expansion
If your new domain uses specialized vocabulary (e.g., Medical, Korean):
1.  Train a new tokenizer on the target corpus.
2.  Merge it with the base model tokenizer.
3.  Resize model embeddings: `model.resize_token_embeddings(len(new_tokenizer))`.
4.  **Important**: Initialize new tokens with the average embedding of existing tokens, not random noise, to speed up convergence.

## 3. Learning Rate
* Use a **lower LR** than the original pre-training max LR.
* Typically 5% to 10% of the original max LR.
* Use a cosine schedule with warmup.
""",

    "references/02_fine_tuning/sft_best_practices.md": """# Supervised Fine-Tuning (SFT) Checklist

## 1. Sequence Packing (`packing=True`)
* **Concept**: Concatenates samples to fill the `max_seq_length`.
* **Impact**: 2x-3x training speedup.
* **Code**:
    ```python
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        packing=True,
        max_seq_length=4096,
        dataset_text_field="text" # Dataset must be pre-formatted strings
    )
    ```

## 2. Chat Templates
Never manually concatenate strings. Use the tokenizer's template.
```python
def apply_chat_template(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"], 
            tokenize=False, 
            add_generation_prompt=False
        )
    }
```

* **Verification**: Run `scripts/stream_data_preview.py` to ensure the template renders special tokens (e.g., `<|im_start|>`) correctly.

## 3. Loss Masking

Ensure you are using `DataCollatorForCompletionOnlyLM`. The model should **not** learn to predict the user's instructions, only the assistant's responses.

```python
response_template = "<|im_start|>assistant\\n"
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template, 
    tokenizer=tokenizer
)
```
""",

    "references/02_fine_tuning/peft_lora_config.md": """# PEFT & LoRA Configuration

## Recommended Settings for 7B-14B Models
Based on community benchmarks and SmolLM findings.

```python
from peft import LoraConfig

peft_config = LoraConfig(
    r=16,                  # Rank: 16-64 is usually sufficient
    lora_alpha=32,         # Alpha: Usually 2 * rank
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear", # Crucial: target all linear layers, not just q_proj, v_proj
    use_dora=True          # DoRA usually outperforms standard LoRA slightly
)
```

## QLoRA (Quantized LoRA)

* Load base model in 4-bit (`load_in_4bit=True` in `BitsAndBytesConfig`).
* Train the adapter in BF16/FP16.
* **VRAM Impact**: Allows fine-tuning a 70B model on two 24GB GPUs (e.g., 3090/4090).
""",

    "references/03_alignment/dpo_orpo_preference.md": """# Preference Optimization: DPO vs ORPO

Moving beyond SFT to align with human preferences.

## 1. DPO (Direct Preference Optimization)
* **Status**: Industry standard. Stable and memory efficient.
* **Requires**: SFT Model + Reference Model (usually the same SFT model frozen).
* **Data**: Triplets `(prompt, chosen, rejected)`.
* **Key Param**: `beta` (0.1 is standard). Higher beta = more conservative (closer to ref model).

## 2. ORPO (Odds Ratio Preference Optimization)
* **Status**: Newer method. Merges SFT and Alignment into one stage.
* **Pros**: No Reference Model needed (saved VRAM).
* **Cons**: Slightly more sensitive to hyperparameters.
* **Use Case**: When you want to train from a Base model directly to Aligned model using preference data, skipping a separate SFT stage.

## 3. SimPO (Simple Preference Optimization)
* **Idea**: Like DPO but uses the average log probability of the sequence as the reward, normalized by length.
* **Pros**: Helps prevent the model from gaming the reward by just generating longer responses.
""",

    "references/03_alignment/grpo_reasoning.md": """# GRPO: Reasoning Alignment (DeepSeek-R1 Style)

**Group Relative Policy Optimization (GRPO)** is designed for Reinforcement Learning (RL) on tasks with verifiable outcomes (Math, Code) without the massive overhead of a Critic model.

## Core Mechanics
1.  **Group Sampling**: For one prompt, sample $G$ outputs (e.g., 8 or 16).
2.  **Relative Reward**: Score all outputs. Calculate advantage based on the group mean/std.
    * $Adv_i = \\frac{Score_i - Mean(GroupScore)}{Std(GroupScore)}$
3.  **Policy Update**: Maximize likelihood of high-advantage outputs.

## Configuration in TRL
```python
from trl import GRPOTrainer, GRPOConfig

training_args = GRPOConfig(
    num_generations=8,           # The 'Group' size
    max_completion_length=1024,
    use_vllm=True,               # Critical for speed!
    beta=0.04                    # KL penalty
)

# Reward Function: Strict Format + Correctness
def reward_correctness(prompts, completions, answer, **kwargs):
    # logic to check if answer is in completion
    return [1.0 if a in c else 0.0 for a, c in zip(answer, completions)]
```

## Why use vLLM?

GRPO generates data *online* during training. Generating 8 sequences for batch size 4 = 32 sequences per step. Without vLLM optimization (paged attention), the generation step will be 10x slower than the training step.
""",

    "references/04_distributed/strategy_matrix.md": """# Distributed Training Strategy Matrix

| Scenario | Strategy | Accelerate Config | Note |
| :--- | :--- | :--- | :--- |
| **1 GPU** (Fits VRAM) | Standard | `default` | Easiest debugging. |
| **Multi-GPU** (Fits on 1) | DDP | `multi_gpu` | Fastest multi-gpu. Replicates model. |
| **Multi-GPU** (Doesn't fit) | ZeRO-2 | `deepspeed` | Shards optimizer/gradients. Moderate comms. |
| **Multi-GPU** (Large Model) | ZeRO-3 / FSDP | `deepspeed` / `fsdp` | Shards parameters. High comms. Essential for 70B+. |
| **VRAM Starved** | ZeRO-3 Offload | `deepspeed` | Offloads to CPU RAM. Slow, but handles massive models. |

## Quick Config Command
```bash
accelerate config
# Select 'This machine', 'Multi-GPU', then choose 'DeepSpeed' or 'FSDP'
```
""",

    "references/04_distributed/deepspeed_fsdp_config.md": """# DeepSpeed vs FSDP Configuration

## DeepSpeed (ZeRO-3)
Best for pushing the limits of model size.
* **config.json key**: `zero_optimization.stage = 3`
* **overlap_comm**: True (Overlaps communication with computation).
* **offload_optimizer**: `{"device": "cpu"}` (Saves tons of VRAM, costs PCIe bandwidth).

## FSDP (PyTorch Native)
Best for Llama/Mistral architectures within PyTorch ecosystem.
* **Sharding Strategy**: `FULL_SHARD` (Equivalent to ZeRO-3).
* **Auto Wrap Policy**: `TRANSFORMER_BASED_WRAP`.
* **Mixed Precision**: `BF16` (Mandatory for stability on Ampere+).

**Tip**: Always save checkpoints as `SHARDED_STATE_DICT` to avoid OOM when saving a massive model on rank 0. Use `merge_checkpoint` scripts later to combine them.
""",

    "references/05_inference/production_optimization.md": """# Inference Optimization

## 1. Throughput vs Latency
* **Latency**: Time to first token (TTFT). Optimized by lower precision and faster interconnects.
* **Throughput**: Tokens per second. Optimized by batching.

## 2. Flash Attention 2
The single most important optimization.
* Reduces memory complexity from $O(N^2)$ to linear.
* **Usage**: `model = AutoModel.from_pretrained(..., attn_implementation="flash_attention_2")`.

## 3. Quantization
* **AWQ/GPTQ**: Weight-only quantization. Good for compute-bound scenarios (batch size 1).
* **KV Cache Quantization (FP8)**: Crucial for long-context inference. Reduces cache memory by 2x, allowing 2x larger batch sizes.

## 4. Serving Engines (Do not use raw PyTorch loop)
* **vLLM**: State-of-the-art PagedAttention. Best for throughput.
* **TGI (Text Generation Inference)**: Hugging Face standard. Great Tensor Parallelism support.
* **SGLang**: Best for structured output (JSON enforcement).
""",
}


def main():
    print("Creating hf-llm-architect skill structure...")
    
    # Create directories
    for dir_path in DIRS:
        full_path = BASE_DIR / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {full_path}")
    
    # Create files
    for file_path, content in FILES.items():
        full_path = BASE_DIR / file_path
        full_path.write_text(content)
        # Make scripts executable
        if full_path.parent.name == "scripts":
            full_path.chmod(0o755)
        print(f"✓ Created file: {full_path}")
    
    print(f"\\n✅ Successfully created hf-llm-architect skill at {BASE_DIR}")
    print("\\nNext steps:")
    print("1. Review the created files")
    print("2. Test the scripts with: cd skills/hf-llm-architect && ./scripts/env_health_check.py")
    print("3. Commit the changes to git")


if __name__ == "__main__":
    main()
