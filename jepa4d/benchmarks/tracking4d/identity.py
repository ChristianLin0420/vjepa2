"""Labeled two-instance crossing, occlusion, and re-entry fixture with identity metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from jepa4d.data.rgb_input import from_view_sequences
from jepa4d.data.schemas import JEPATokenBundle, RGBInputBatch
from jepa4d.models.object_slot_grounder import ObjectObservation, ObjectSlot, ObjectSlotGrounder

FeatureSource = Literal["oracle", "ambiguous", "rgb", "vjepa", "vjepa_mask"]


@dataclass(slots=True)
class IdentityFixture:
    batch: RGBInputBatch
    boxes: dict[tuple[str, int], list[float]]
    visible: dict[tuple[str, int], bool]


@dataclass(slots=True)
class DavisIdentityFixture:
    batch: RGBInputBatch
    masks: list[np.ndarray]
    sequence: str
    source_frame_names: list[str]


def build_crossing_fixture(steps: int = 10, height: int = 128, width: int = 192) -> IdentityFixture:
    """Render two similar instances that cross, disappear, and re-enter."""
    if steps < 8:
        raise ValueError("identity fixture requires at least eight frames")
    frames: list[np.ndarray] = []
    boxes: dict[tuple[str, int], list[float]] = {}
    visible: dict[tuple[str, int], bool] = {}
    y, x = np.mgrid[:height, :width]
    for step in range(steps):
        frame = np.stack(
            (
                25 + (x // 10) % 12,
                30 + (y // 10) % 12,
                35 + ((x + y) // 20) % 12,
            ),
            axis=-1,
        ).astype(np.uint8)
        positions = {"A": 18 + 14 * step, "B": width - 42 - 14 * step}
        visibility = {"A": step not in {3, 4}, "B": step != 7}
        colors = {"A": (190, 45, 45), "B": (170, 55, 55)}
        for identity in ("A", "B"):
            x1 = int(np.clip(positions[identity], 0, width - 24))
            y1 = 45 if identity == "A" else 58
            box = [float(x1), float(y1), float(x1 + 24), float(y1 + 30)]
            boxes[(identity, step)] = box
            visible[(identity, step)] = visibility[identity]
            if visibility[identity]:
                frame[y1 : y1 + 30, x1 : x1 + 24] = colors[identity]
                frame[y1 + 5 : y1 + 10, x1 + 7 : x1 + 17] = (225, 220, 210)
        frames.append(frame)
    return IdentityFixture(batch=from_view_sequences([frames]), boxes=boxes, visible=visible)


def fixture_observations(
    fixture: IdentityFixture,
    grounder: ObjectSlotGrounder,
    *,
    feature_source: FeatureSource,
    tokens: JEPATokenBundle | None = None,
) -> list[ObjectObservation]:
    if feature_source in {"vjepa", "vjepa_mask"} and tokens is None:
        raise ValueError("V-JEPA feature source requires a token bundle")
    observations: list[ObjectObservation] = []
    height, width = fixture.batch.images.shape[-2:]
    for time_index in range(fixture.batch.images.shape[2]):
        for identity in ("A", "B"):
            if not fixture.visible[(identity, time_index)]:
                continue
            bbox = fixture.boxes[(identity, time_index)]
            mask = grounder._box_mask(bbox, height, width)
            if feature_source == "oracle":
                embedding = np.asarray([1.0, 0.0, 0.1] if identity == "A" else [0.0, 1.0, 0.1])
                embedding = embedding / np.linalg.norm(embedding)
            elif feature_source == "ambiguous":
                embedding = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                embedding = grounder.extract_visual_embedding(
                    fixture.batch.images[0, 0, time_index],
                    bbox,
                    "mug",
                    tokens if feature_source in {"vjepa", "vjepa_mask"} else None,
                    0,
                    time_index,
                    mask=mask if feature_source == "vjepa_mask" else None,
                )
            physical_x = (bbox[0] + bbox[2]) / (2 * width) * 2.0
            observations.append(
                ObjectObservation(
                    observation_id=f"gt-{identity}-t{time_index:02d}",
                    batch_index=0,
                    view_index=0,
                    time_index=time_index,
                    camera_id="camera_0",
                    category="mug",
                    score=0.9,
                    bbox_2d=bbox,
                    mask=mask,
                    visual_embedding=np.asarray(embedding, dtype=np.float32),
                    pose_map=[physical_x, bbox[1] / height, 1.0],
                )
            )
    return observations


def load_davis_fixture(
    root: str | Path,
    *,
    sequence: str = "dogs-jump",
    stride: int = 3,
    max_frames: int = 24,
) -> DavisIdentityFixture:
    from PIL import Image

    root = Path(root)
    image_paths = sorted((root / "JPEGImages" / "480p" / sequence).glob("*.jpg"))[::stride][:max_frames]
    if not image_paths:
        raise FileNotFoundError(f"no DAVIS frames found for {sequence!r} under {root}")
    frames = [np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).copy() for path in image_paths]
    masks = [
        np.asarray(Image.open(root / "Annotations" / "480p" / sequence / f"{path.stem}.png")) for path in image_paths
    ]
    return DavisIdentityFixture(
        batch=from_view_sequences([frames]),
        masks=masks,
        sequence=sequence,
        source_frame_names=[path.name for path in image_paths],
    )


def davis_observations(
    fixture: DavisIdentityFixture,
    grounder: ObjectSlotGrounder,
    *,
    feature_source: FeatureSource,
    tokens: JEPATokenBundle | None = None,
    minimum_mask_area: int = 64,
) -> list[ObjectObservation]:
    if feature_source in {"vjepa", "vjepa_mask"} and tokens is None:
        raise ValueError("V-JEPA feature source requires a token bundle")
    identities = sorted({int(value) for mask in fixture.masks for value in np.unique(mask) if value > 0})
    observations: list[ObjectObservation] = []
    for time_index, annotation in enumerate(fixture.masks):
        for identity_index, identity in enumerate(identities):
            mask = annotation == identity
            if int(mask.sum()) < minimum_mask_area:
                continue
            ys, xs = np.where(mask)
            bbox = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
            if feature_source == "oracle":
                embedding = np.zeros(len(identities), dtype=np.float32)
                embedding[identity_index] = 1.0
            elif feature_source == "ambiguous":
                embedding = np.ones(1, dtype=np.float32)
            else:
                embedding = grounder.extract_visual_embedding(
                    fixture.batch.images[0, 0, time_index],
                    bbox,
                    "dog",
                    tokens if feature_source in {"vjepa", "vjepa_mask"} else None,
                    0,
                    time_index,
                    mask=mask if feature_source == "vjepa_mask" else None,
                )
            observations.append(
                ObjectObservation(
                    observation_id=f"gt-{identity}-t{time_index:02d}",
                    batch_index=0,
                    view_index=0,
                    time_index=time_index,
                    camera_id="camera_0",
                    category="dog",
                    score=1.0,
                    bbox_2d=bbox,
                    mask=mask,
                    visual_embedding=np.asarray(embedding, dtype=np.float32),
                )
            )
    return observations


def identity_metrics(slots: list[ObjectSlot]) -> dict[str, float]:
    assignment: dict[str, str] = {}
    truth: dict[str, str] = {}
    for slot in slots:
        for reference in slot.observation_refs:
            assignment[reference] = slot.object_id
            truth[reference] = reference.split("-")[1]
    references = sorted(assignment)
    true_positive = false_positive = false_negative = 0
    for first_index, first in enumerate(references):
        for second in references[first_index + 1 :]:
            same_truth = truth[first] == truth[second]
            same_prediction = assignment[first] == assignment[second]
            true_positive += int(same_truth and same_prediction)
            false_positive += int(not same_truth and same_prediction)
            false_negative += int(same_truth and not same_prediction)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    identity_tracks: dict[str, list[tuple[int, str]]] = {identity: [] for identity in sorted(set(truth.values()))}
    for reference, track_id in assignment.items():
        identity = truth[reference]
        time_index = int(reference.rsplit("t", 1)[1])
        identity_tracks[identity].append((time_index, track_id))
    switches = 0
    fragments = 0
    survival = []
    for values in identity_tracks.values():
        ordered = [track_id for _, track_id in sorted(values)]
        switches += sum(first != second for first, second in zip(ordered, ordered[1:], strict=False))
        counts = {track_id: ordered.count(track_id) for track_id in set(ordered)}
        fragments += max(0, len(counts) - 1)
        survival.append(max(counts.values(), default=0) / max(len(ordered), 1))
    false_merges = sum(len({reference.split("-")[1] for reference in slot.observation_refs}) > 1 for slot in slots)
    return {
        "pairwise_precision": precision,
        "pairwise_recall": recall,
        "pairwise_f1": f1,
        "id_switches": float(switches),
        "fragments": float(fragments),
        "false_merges": float(false_merges),
        "track_survival": float(np.mean(survival)) if survival else 0.0,
        "predicted_tracks": float(len(slots)),
        "observations": float(len(references)),
    }


def run_davis_variant(
    fixture: DavisIdentityFixture,
    *,
    feature_source: FeatureSource,
    tokens: JEPATokenBundle | None,
    weights: tuple[float, float, float],
    threshold: float,
    max_time_gap: int = 4,
) -> tuple[list[ObjectSlot], dict[str, float]]:
    grounder = ObjectSlotGrounder(
        appearance_weight=weights[0],
        iou_weight=weights[1],
        geometry_weight=weights[2],
        association_threshold=threshold,
        max_time_gap=max_time_gap,
    )
    observations = davis_observations(fixture, grounder, feature_source=feature_source, tokens=tokens)
    slots = grounder.associate_observations(observations, fixture.batch)
    return slots, identity_metrics(slots)


def run_identity_variant(
    fixture: IdentityFixture,
    *,
    feature_source: FeatureSource,
    tokens: JEPATokenBundle | None,
    weights: tuple[float, float, float],
    threshold: float,
    max_time_gap: int = 4,
) -> tuple[list[ObjectSlot], dict[str, float]]:
    grounder = ObjectSlotGrounder(
        appearance_weight=weights[0],
        iou_weight=weights[1],
        geometry_weight=weights[2],
        association_threshold=threshold,
        max_time_gap=max_time_gap,
    )
    observations = fixture_observations(fixture, grounder, feature_source=feature_source, tokens=tokens)
    slots = grounder.associate_observations(observations, fixture.batch)
    return slots, identity_metrics(slots)
