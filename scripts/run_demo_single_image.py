"""Run a zero-download single-image Phase 1 demo."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--model", default="vjepa2_1_vit_base_384")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo_single_image"))
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    image = args.image or args.output / "synthetic_kitchen.png"
    if args.image is None:
        y, x = np.mgrid[:384, :384]
        value = np.stack(((x * 2) % 256, (y * 3) % 256, ((x + y) // 2) % 256), axis=-1).astype(np.uint8)
        Image.fromarray(value).save(image)
    command = [
        sys.executable,
        "-m",
        "jepa4d.cli.encode",
        "--input",
        str(image),
        "--output",
        str(args.output / "features.pt"),
        "--model",
        args.model,
    ]
    if args.mock:
        command.append("--mock")
    if args.checkpoint:
        command.extend(("--checkpoint", str(args.checkpoint)))
    if args.wandb:
        command.append("--wandb")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
