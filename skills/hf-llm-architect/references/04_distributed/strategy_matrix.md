# Distributed Training Strategy Matrix

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
