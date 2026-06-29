"""CPU-safe Phase 3 object grounding and memory demo."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector-backend", choices=("mock", "grounding_dino"), default="mock")
    parser.add_argument("--mask-backend", choices=("box", "sam2"), default="box")
    parser.add_argument("--output", type=Path, default=Path("outputs/demo_object_grounding"))
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    input_dir = args.output / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    y, x = np.mgrid[:256, :384]
    images = []
    for index in range(2):
        image = np.zeros((256, 384, 3), dtype=np.uint8)
        image[..., 0] = (x // 2 + index * 15) % 256
        image[..., 1] = (y + index * 10) % 256
        image[90:180, 70 + 12 * index : 150 + 12 * index] = (220, 30, 30)
        image[130:220, 220 - 8 * index : 350 - 8 * index] = (100, 70, 40)
        path = input_dir / f"view_{index}.png"
        Image.fromarray(image).save(path)
        images.append(path)
    command = [sys.executable, "-m", "jepa4d.cli.build_memory"]
    for path in images:
        command.extend(("--images", str(path)))
    command.extend(
        (
            "--query",
            "red mug",
            "--query",
            "wooden table",
            "--detector-backend",
            args.detector_backend,
            "--mask-backend",
            args.mask_backend,
            "--output",
            str(args.output),
        )
    )
    if args.wandb:
        command.append("--wandb")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
