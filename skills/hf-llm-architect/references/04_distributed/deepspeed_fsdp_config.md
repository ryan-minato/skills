# DeepSpeed vs FSDP Configuration

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
