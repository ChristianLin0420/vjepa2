"""Mock video encoding plus explicit memory update demonstration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jepa4d.data.rgb_input import from_view_sequences
from jepa4d.memory.memory_update import FourDMemoryCore
from jepa4d.models.vjepa21_adapter import VJEPA21FeatureExtractor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", default=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo_video_memory"))
    args = parser.parse_args()
    y, x = np.mgrid[:192, :256]
    frames = [
        np.stack(((x + 13 * i) % 256, (y + 7 * i) % 256, ((x + y) // 2 + 17 * i) % 256), axis=-1).astype(np.uint8)
        for i in range(8)
    ]
    batch = from_view_sequences([frames])
    bundle = VJEPA21FeatureExtractor(mock=True)(batch)
    memory = FourDMemoryCore()
    memory.active_local_map.observations.append(
        {
            "mode": batch.mode,
            "token_shape": list(bundle.dense_tokens.shape),
            "temporal_bins": bundle.dense_tokens.shape[2],
        }
    )
    args.output.mkdir(parents=True, exist_ok=True)
    bundle.save(args.output / "features.pt")
    (args.output / "memory.json").write_text(
        json.dumps(
            {
                "active_local_map": memory.active_local_map.observations,
                "scene_graph": memory.scene_graph.to_serializable(),
            },
            indent=2,
        )
    )
    (args.output / "EXPERIMENT.md").write_text(
        f"# Mock video-memory experiment\n\n- Input frames: 8\n- Token shape: `{list(bundle.dense_tokens.shape)}`\n- Result: memory updated successfully\n"
    )
    print(args.output / "memory.json")


if __name__ == "__main__":
    main()
