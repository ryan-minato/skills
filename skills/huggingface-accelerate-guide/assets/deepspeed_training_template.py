# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "accelerate>=1.1.0",
# ]
# ///

import argparse
import json
from pathlib import Path

from accelerate import Accelerator


def load_ds_config(path: str) -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"DeepSpeed config not found: {path}")
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSpeed-friendly Accelerate bootstrap")
    parser.add_argument("--deepspeed_config", type=str, default="")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--with_tracking", action="store_true")
    args = parser.parse_args()

    if args.deepspeed_config:
        ds_cfg = load_ds_config(args.deepspeed_config)
        print({"deepspeed_keys": sorted(ds_cfg.keys())})

    accelerator = (
        Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, log_with="tensorboard")
        if args.with_tracking
        else Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    )
    accelerator.print(f"distributed_type={accelerator.distributed_type}")
    accelerator.print(f"gradient_accumulation_steps={accelerator.gradient_accumulation_steps}")
    accelerator.print("Recommended command: accelerate launch --config_file <accelerate_config.yaml> deepspeed_training_template.py")


if __name__ == "__main__":
    main()
