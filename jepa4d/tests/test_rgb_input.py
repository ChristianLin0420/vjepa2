import numpy as np
import pytest
import torch

from jepa4d.data.rgb_input import collate_rgb_inputs, from_view_sequences
from jepa4d.data.schemas import RGBInputBatch
from jepa4d.models.viewset_tokenizer import ViewSetTokenizer


def frame(value: int = 0) -> np.ndarray:
    return np.full((32, 48, 3), value, dtype=np.uint8)


@pytest.mark.parametrize(
    ("views", "steps", "mode"),
    [(1, 1, "single_image"), (3, 1, "multi_view"), (1, 4, "video"), (2, 3, "multiview_video")],
)
def test_modes_are_explicit(views: int, steps: int, mode: str) -> None:
    batch = from_view_sequences([[frame(v + t) for t in range(steps)] for v in range(views)])
    assert batch.shape == (1, views, steps, 3, 32, 48)
    assert batch.mode == mode
    assert batch.valid_mask.all()


def test_invalid_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="disagrees"):
        RGBInputBatch(torch.zeros(1, 1, 1, 3, 8, 8), torch.zeros(1, 1, 1), [["camera"]], "video")


def test_collate_pads_and_masks() -> None:
    single = from_view_sequences([[frame()]])
    video = from_view_sequences([[frame(), frame(1), frame(2)]])
    batch = collate_rgb_inputs([single, video])
    assert batch.images.shape == (2, 1, 3, 3, 32, 48)
    assert batch.valid_mask[0].sum() == 1
    assert batch.valid_mask[1].sum() == 3


def test_viewset_tokenizer_preserves_identity() -> None:
    batch = from_view_sequences([[frame()], [frame(1)]])
    encoded = ViewSetTokenizer(embed_dim=16)(batch)
    assert encoded.identity_tokens.shape == (1, 2, 1, 16)
    assert not torch.equal(encoded.identity_tokens[:, 0], encoded.identity_tokens[:, 1])
