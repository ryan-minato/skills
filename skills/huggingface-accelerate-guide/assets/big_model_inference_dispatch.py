# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "accelerate>=1.1.0",
#   "torch>=2.3.0",
# ]
# ///

import argparse
from pathlib import Path

import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="Big model inference template via load_checkpoint_and_dispatch")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--device_map", type=str, default="auto")
    args = parser.parse_args()

    with init_empty_weights():
        model = TinyModel()

    checkpoint = Path(args.checkpoint)
    if not args.checkpoint or not checkpoint.exists():
        print(
            "Please pass --checkpoint pointing to a weights file/folder. "
            "This script demonstrates the init_empty_weights + load_checkpoint_and_dispatch pattern."
        )
        return

    model = load_checkpoint_and_dispatch(model, checkpoint=str(checkpoint), device_map=args.device_map)
    first_device = next(model.parameters()).device
    x = torch.randn(2, 8, device=first_device)
    with torch.no_grad():
        y = model(x)
    print({"device": str(first_device), "output_shape": tuple(y.shape)})


if __name__ == "__main__":
    main()
