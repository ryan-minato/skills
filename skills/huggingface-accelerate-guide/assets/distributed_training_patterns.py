# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "accelerate>=1.1.0",
#   "torch>=2.3.0",
# ]
# ///

import argparse

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import DDPCommunicationHookType, DistributedDataParallelKwargs


def build_accelerator(mode: str) -> Accelerator:
    if mode == "ddp-hook":
        ddp_kwargs = DistributedDataParallelKwargs(
            comm_hook=DDPCommunicationHookType.FP16,
        )
        return Accelerator(kwargs_handlers=[ddp_kwargs])
    if mode == "fsdp":
        fsdp_plugin = FullyShardedDataParallelPlugin()
        return Accelerator(fsdp_plugin=fsdp_plugin)
    return Accelerator()


def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed training patterns scaffold")
    parser.add_argument("--mode", choices=["baseline", "ddp-hook", "fsdp"], default="baseline")
    args = parser.parse_args()

    accelerator = build_accelerator(args.mode)
    state = accelerator.state
    accelerator.print(f"mode={args.mode}")
    accelerator.print(f"distributed_type={state.distributed_type}")
    accelerator.print(f"num_processes={state.num_processes}")
    accelerator.print("use `accelerate launch` with multi-GPU/multi-node config to activate distributed features")


if __name__ == "__main__":
    main()
