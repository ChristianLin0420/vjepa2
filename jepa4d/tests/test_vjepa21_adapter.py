from pathlib import Path

import numpy as np
import pytest
import torch

from jepa4d.data.rgb_input import from_view_sequences
from jepa4d.models.vjepa21_adapter import VJEPA21FeatureExtractor

PHASE2B_ASSETS = Path("checkpoints/phase2b_assets")
LOCAL_CHECKPOINT = PHASE2B_ASSETS / "vjepa2.1-vitb-fpc64-384"
LOCAL_IMPLEMENTATION = PHASE2B_ASSETS / "vjepa21_hf_impl"


def frames(count: int) -> list[np.ndarray]:
    return [np.full((48, 64, 3), index * 10, dtype=np.uint8) for index in range(count)]


@pytest.mark.parametrize(("count", "expected_steps", "modality"), [(1, 1, "image"), (4, 2, "video"), (5, 3, "video")])
def test_mock_shapes(count: int, expected_steps: int, modality: str) -> None:
    batch = from_view_sequences([frames(count)])
    extractor = VJEPA21FeatureExtractor(mock=True, mock_embed_dim=32)
    first = extractor(batch)
    second = extractor(batch)
    assert first.dense_tokens.shape == (1, 1, expected_steps, 576, 32)
    assert first.global_tokens.shape == (1, 1, expected_steps, 32)
    assert first.modality == modality
    assert torch.equal(first.dense_tokens, second.dense_tokens)
    assert torch.isfinite(first.dense_tokens).all()


def test_mock_multiview_shape() -> None:
    batch = from_view_sequences([[frames(1)[0]], [frames(1)[0]], [frames(1)[0]]])
    bundle = VJEPA21FeatureExtractor(mock=True)(batch)
    assert bundle.dense_tokens.shape[:3] == (1, 3, 1)


def test_mock_can_disable_intermediate_capture() -> None:
    batch = from_view_sequences([frames(1)])
    bundle = VJEPA21FeatureExtractor(mock=True, capture_layers=())(batch)
    assert bundle.layer_tokens == {}
    assert bundle.metadata["model"]["captured_layers"] == []


@pytest.mark.skipif(
    not (LOCAL_CHECKPOINT / "model.safetensors").is_file()
    or not (LOCAL_IMPLEMENTATION / "modeling_vjepa21.py").is_file(),
    reason="matched local model and implementation assets are absent",
)
def test_local_real_checkpoint() -> None:
    batch = from_view_sequences([frames(1)])
    extractor = VJEPA21FeatureExtractor(
        checkpoint=LOCAL_CHECKPOINT,
        implementation_path=LOCAL_IMPLEMENTATION,
    )
    bundle = extractor(batch)
    assert bundle.dense_tokens.shape == (1, 1, 1, 576, 768)
    assert sorted(bundle.layer_tokens) == [2, 5, 8, 11]
    assert torch.isfinite(bundle.dense_tokens).all()
