# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "accelerate>=1.1.0",
# ]
# ///

import argparse

from accelerate import Accelerator
from accelerate.utils import AORecipeKwargs, MSAMPRecipeKwargs, TERecipeKwargs


def pick_handler(backend: str):
    if backend == "msamp":
        return MSAMPRecipeKwargs(optimization_level="O1")
    if backend == "te":
        return TERecipeKwargs()
    if backend == "ao":
        return AORecipeKwargs()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="FP8 mixed precision template")
    parser.add_argument("--backend", choices=["auto", "msamp", "te", "ao"], default="auto")
    args = parser.parse_args()

    handler = pick_handler(args.backend)
    kwargs = [handler] if handler is not None else None

    try:
        accelerator = Accelerator(mixed_precision="fp8", kwarg_handlers=kwargs)
        accelerator.print({"mixed_precision": accelerator.mixed_precision, "backend": args.backend})
    except Exception as exc:
        print({"status": "fp8 init failed", "reason": str(exc), "hint": "Check GPU architecture and backend installation."})


if __name__ == "__main__":
    main()
