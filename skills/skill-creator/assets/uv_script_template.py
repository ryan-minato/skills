#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   # "package>=X.Y,<X+1",
# ]
# requires-python = ">=3.11"
# ///

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="[One-line description.]",
        epilog="Example: uv run scripts/[name].py --flag VALUE",
    )
    # Add arguments here.
    # parser.add_argument("--flag", required=True, help="[description]")
    args = parser.parse_args()

    # --- implementation ---

    # Data → stdout (JSON preferred for machine consumption).
    # print(json.dumps(result, indent=2))

    # Diagnostics → stderr.
    # print("message", file=sys.stderr)

    # Exit codes: sys.exit(0) success, sys.exit(1) error, sys.exit(2) bad args.


if __name__ == "__main__":
    main()
