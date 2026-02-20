# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "accelerate>=1.1.0",
# ]
# ///

import argparse

from accelerate import PartialState


def main() -> None:
    parser = argparse.ArgumentParser(description="Split prompts across processes for distributed inference")
    parser.add_argument("--prompts", nargs="+", default=["a robot", "a tiger", "a city at night"])
    parser.add_argument("--apply_padding", action="store_true")
    args = parser.parse_args()

    state = PartialState()
    with state.split_between_processes(args.prompts, apply_padding=args.apply_padding) as prompt_chunk:
        local_outputs = [f"proc{state.process_index}:{p}" for p in prompt_chunk]

    print({
        "process_index": state.process_index,
        "num_processes": state.num_processes,
        "local_outputs": local_outputs,
        "note": "When gathering with padding enabled, drop the duplicated last sample after gather.",
    })


if __name__ == "__main__":
    main()
