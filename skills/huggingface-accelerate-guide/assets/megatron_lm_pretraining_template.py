# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "accelerate>=1.1.0",
# ]
# ///

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Megatron-LM launch planning template")
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--num_processes", type=int, default=4)
    parser.add_argument("--script", type=str, default="examples/by_feature/megatron_lm_gpt_pretraining.py")
    args = parser.parse_args()

    required = args.tp * args.pp
    if required > args.num_processes:
        raise ValueError(f"tp*pp={required} must be <= num_processes={args.num_processes}")

    cmd = (
        f"accelerate launch --num_processes {args.num_processes} {args.script} "
        f"--megatron_lm_tp_degree {args.tp} --megatron_lm_pp_degree {args.pp}"
    )
    print({"required_processes": required, "launch_command": cmd})


if __name__ == "__main__":
    main()
