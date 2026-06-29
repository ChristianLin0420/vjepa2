"""Run a zero-download multi-view Phase 1 demo."""

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
    parser.add_argument("--images", nargs="*", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo_multiview"))
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    images = args.images or []
    if not images:
        y, x = np.mgrid[:384, :384]
        for index in range(3):
            path = args.output / f"synthetic_view_{index}.png"
            value = np.stack(((x + 30 * index) % 256, (y + 50 * index) % 256, ((x + y) // 2) % 256), axis=-1).astype(
                np.uint8
            )
            Image.fromarray(value).save(path)
            images.append(path)
    command = [sys.executable, "-m", "jepa4d.cli.encode"]
    for image in images:
        command.extend(("--input", str(image)))
    command.extend(("--output", str(args.output / "features.pt"), "--model", args.model))
    if args.mock:
        command.append("--mock")
    if args.checkpoint:
        command.extend(("--checkpoint", str(args.checkpoint)))
    if args.wandb:
        command.append("--wandb")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
