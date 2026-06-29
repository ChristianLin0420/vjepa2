import numpy as np

from jepa4d.benchmarks.tracking4d.identity import (
    build_crossing_fixture,
    identity_metrics,
    run_identity_variant,
)
from jepa4d.models.object_slot_grounder import ObjectObservation, ObjectSlotGrounder


def test_same_frame_detections_are_exclusive() -> None:
    fixture = build_crossing_fixture(steps=8)
    mask = np.ones((128, 192), dtype=bool)
    observations = [
        ObjectObservation(
            observation_id=f"gt-{identity}-t00",
            batch_index=0,
            view_index=0,
            time_index=0,
            camera_id="camera_0",
            category="mug",
            score=0.9,
            bbox_2d=[10.0, 10.0, 30.0, 30.0],
            mask=mask,
            visual_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        )
        for identity in ("A", "B")
    ]
    slots = ObjectSlotGrounder(association_threshold=0.0).associate_observations(observations, fixture.batch)
    assert len(slots) == 2
    assert all(len(slot.observations) == 1 for slot in slots)


def test_controlled_identity_metrics_detect_merges_and_switches() -> None:
    fixture = build_crossing_fixture()
    _, oracle = run_identity_variant(
        fixture,
        feature_source="oracle",
        tokens=None,
        weights=(1.0, 0.0, 0.0),
        threshold=0.8,
    )
    _, no_appearance = run_identity_variant(
        fixture,
        feature_source="ambiguous",
        tokens=None,
        weights=(0.0, 0.57, 0.43),
        threshold=0.45,
    )
    assert oracle["pairwise_f1"] == 1.0
    assert oracle["id_switches"] == oracle["false_merges"] == 0.0
    assert no_appearance["pairwise_f1"] < oracle["pairwise_f1"]
    assert no_appearance["id_switches"] > 0


def test_empty_identity_metrics_are_finite() -> None:
    metrics = identity_metrics([])
    assert metrics["pairwise_f1"] == 0.0
    assert metrics["track_survival"] == 0.0
