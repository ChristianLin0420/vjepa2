from pathlib import Path

import numpy as np
import pytest

from jepa4d.data.rgb_input import collate_rgb_inputs, from_view_sequences
from jepa4d.models.geometry_belief import GeometryBeliefHead
from jepa4d.models.object_slot_grounder import ObjectSlotGrounder
from jepa4d.models.vjepa21_adapter import VJEPA21FeatureExtractor


def image(offset: int = 0) -> np.ndarray:
    y, x = np.mgrid[:48, :64]
    return np.stack(((x + offset) % 256, (y + offset) % 256, (x + y + offset) % 256), axis=-1).astype(np.uint8)


def test_mock_grounder_associates_views_and_is_deterministic() -> None:
    batch = from_view_sequences([[image()], [image(5)]])
    grounder = ObjectSlotGrounder()
    first = grounder(batch, ["red mug", "wooden table"])
    second = grounder(batch, ["wooden table", "red mug"])

    assert len(first.observations) == 4
    assert len(first.slots) == 2
    assert sorted(slot.object_id for slot in first.slots) == sorted(slot.object_id for slot in second.slots)
    assert all(len(slot.observations) == 2 for slot in first.slots)
    assert all(slot.mask is not None and slot.mask.shape == (48, 64) for slot in first.slots)
    assert all(np.isclose(np.linalg.norm(slot.visual_embedding), 1.0) for slot in first.slots)
    assert first.metadata["mock_outputs_are_not_accuracy_predictions"]


def test_grounder_uses_jepa_tokens_and_geometry() -> None:
    batch = from_view_sequences([[image()], [image(5)]])
    tokens = VJEPA21FeatureExtractor(mock=True)(batch)
    geometry = GeometryBeliefHead(output_size=28)(batch)
    result = ObjectSlotGrounder()(batch, ["mug"], tokens=tokens, geometry=geometry)

    assert result.metadata["uses_jepa_tokens"]
    assert result.metadata["uses_geometry"]
    assert len(result.slots) == 1
    assert result.slots[0].pose_map is not None
    assert len(result.slots[0].pose_map or []) == 3
    assert result.slots[0].confidence["geometry"] > 0


def test_grounding_outputs_are_serializable(tmp_path: Path) -> None:
    result = ObjectSlotGrounder()(from_view_sequences([[image()]]), ["mug"])
    json_path = result.save_json(tmp_path / "objects.json")
    masks_path = result.save_masks(tmp_path / "masks.npz")

    assert '"category": "mug"' in json_path.read_text()
    masks = np.load(masks_path)
    assert masks.files == ["b0-v0-t0-q0"]
    assert masks[masks.files[0]].dtype == np.uint8


def test_empty_query_and_unknown_backends_fail_clearly() -> None:
    batch = from_view_sequences([[image()]])
    with pytest.raises(ValueError, match="non-empty object query"):
        ObjectSlotGrounder()(batch, [" "])
    with pytest.raises(ValueError, match="unknown detector backend"):
        ObjectSlotGrounder(detector_backend="not-a-detector")
    with pytest.raises(ValueError, match="unknown mask backend"):
        ObjectSlotGrounder(mask_backend="not-a-masker")
    with pytest.raises(ValueError, match="expects B=1"):
        ObjectSlotGrounder()(collate_rgb_inputs([batch, batch]), ["mug"])
