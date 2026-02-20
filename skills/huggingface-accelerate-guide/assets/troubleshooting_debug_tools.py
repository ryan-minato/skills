# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "accelerate>=1.1.0",
# ]
# ///

import argparse

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import find_executable_batch_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Troubleshooting helpers: synced logging + auto batch finder")
    parser.add_argument("--starting_batch_size", type=int, default=128)
    parser.add_argument("--max_acceptable_batch_size", type=int, default=32)
    args = parser.parse_args()

    accelerator = Accelerator()
    logger = get_logger(__name__, log_level="INFO")

    @find_executable_batch_size(starting_batch_size=args.starting_batch_size)
    def train_once(batch_size: int) -> int:
        accelerator.free_memory()
        logger.info(f"try batch_size={batch_size}", main_process_only=False, in_order=True)
        if batch_size > args.max_acceptable_batch_size:
            raise RuntimeError("CUDA out of memory.")
        return batch_size

    final_bs = train_once()
    logger.info(f"selected batch_size={final_bs}", main_process_only=False, in_order=True)


if __name__ == "__main__":
    main()
