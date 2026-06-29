"""Download Phase 1 model assets without persisting access tokens."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("checkpoints"))
    parser.add_argument("--repo", default="davevanveen/vjepa2.1-vitb-fpc64-384")
    parser.add_argument("--with-hf-implementation", action="store_true")
    args = parser.parse_args()
    token = os.getenv("HF_TOKEN")
    model_dir = args.output / "vjepa2.1-vitb-fpc64-384"
    print(snapshot_download(args.repo, local_dir=model_dir, token=token))
    if args.with_hf_implementation:
        print(
            snapshot_download(
                "Dev-Jahn/vjepa2.1-vitl-fpc64-384",
                local_dir=args.output / "vjepa21_hf_impl",
                allow_patterns=["*.py"],
                token=token,
            )
        )


if __name__ == "__main__":
    main()
