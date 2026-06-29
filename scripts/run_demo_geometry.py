"""Generate a synthetic multi-view scene and exercise Phase 2 geometry."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("mock", "vggt"), default="mock")
    parser.add_argument("--output", type=Path, default=Path("outputs/demo_geometry"))
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    input_dir = args.output / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    y, x = np.mgrid[:256, :384]
    images = []
    for index in range(3):
        image = np.stack(
            ((x + 24 * index) % 256, (y * 2 + 13 * index) % 256, ((x + y) // 2 + 31 * index) % 256),
            axis=-1,
        ).astype(np.uint8)
        path = input_dir / f"view_{index}.png"
        Image.fromarray(image).save(path)
        images.append(path)
    command = [sys.executable, "-m", "jepa4d.cli.reconstruct"]
    for path in images:
        command.extend(("--images", str(path)))
    command.extend(("--backend", args.backend, "--output", str(args.output)))
    if args.wandb:
        command.append("--wandb")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
