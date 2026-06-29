from pathlib import Path

import numpy as np
import pytest
import torch

from jepa4d.data.rgb_input import from_view_sequences
from jepa4d.models.geometry_belief import GeometryBelief, GeometryBeliefHead, _unproject
from jepa4d.models.geometry_export import export_colmap_text, export_geometry_npz, export_pointcloud_ply


def image(value: int = 0) -> np.ndarray:
    y, x = np.mgrid[:48, :64]
    return np.stack(((x + value) % 256, (y + value) % 256, ((x + y) // 2 + value) % 256), axis=-1).astype(np.uint8)


def test_single_image_is_low_confidence_belief() -> None:
    batch = from_view_sequences([[image()]])
    belief = GeometryBeliefHead(output_size=28)(batch)
    assert belief.scale_confidence.item() <= 0.15
    assert belief.depth_logvar is not None and belief.depth_logvar.mean() > 1.0
    assert belief.depth_mean is not None and belief.depth_mean.shape == (1, 1, 1, 28, 28)
    assert belief.pointmap_mean is not None and belief.pointmap_mean.shape == (1, 1, 1, 28, 28, 3)
    assert belief.tracks_2d is not None and belief.tracks_3d is not None
    assert belief.metadata["synthetic_geometry"]


def test_more_views_increase_belief_without_claiming_certainty() -> None:
    single = from_view_sequences([[image()]])
    multiview = from_view_sequences([[image()], [image(20)], [image(40)]])
    head = GeometryBeliefHead(output_size=28, query_grid_size=4)
    single_belief = head(single)
    multi_belief = head(multiview)
    assert multi_belief.scale_confidence > single_belief.scale_confidence
    assert multi_belief.pose_confidence > single_belief.pose_confidence
    assert multi_belief.scale_confidence.item() < 0.8
    assert multi_belief.tracks_2d is not None and multi_belief.tracks_2d.shape == (1, 3, 16, 2)
    assert multi_belief.tracks_3d is not None and multi_belief.tracks_3d.shape == (1, 3, 16, 3)


def test_known_calibration_and_scale_prior_are_explicit() -> None:
    batch = from_view_sequences([[image()]])
    batch.intrinsics = torch.eye(3).reshape(1, 1, 3, 3)
    belief = GeometryBeliefHead(output_size=28, known_scale_prior=True)(batch)
    assert belief.scale_confidence.item() > 0.5


def test_geometry_exports_are_portable(tmp_path: Path) -> None:
    batch = from_view_sequences([[image()], [image(20)]])
    belief = GeometryBeliefHead(output_size=28)(batch)
    npz = export_geometry_npz(belief, tmp_path / "belief.npz")
    ply = export_pointcloud_ply(belief, batch, tmp_path / "points.ply", max_points=100)
    values = np.load(npz)
    assert values["depth_mean"].shape == (1, 2, 1, 28, 28)
    assert ply.read_text().startswith("ply\nformat ascii 1.0")
    assert "element vertex 100" in ply.read_text().split("end_header")[0]
    colmap = export_colmap_text(belief, batch, tmp_path / "colmap")
    assert "PINHOLE 28 28" in (colmap / "cameras.txt").read_text()
    assert "view_0" not in (colmap / "images.txt").read_text()
    assert (colmap / "points3D.txt").exists()


def test_confidence_range_is_enforced() -> None:
    with pytest.raises(ValueError, match=r"within \[0,1\]"):
        GeometryBelief(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            torch.tensor([1.2]),
            torch.tensor([0.5]),
            torch.tensor([0.5]),
            "invalid",
        )


def test_unprojection_uses_camera_from_world_extrinsics() -> None:
    depth = torch.ones(1, 1, 1, 1)
    intrinsics = torch.eye(3).reshape(1, 1, 3, 3)
    camera_from_world = torch.eye(4).reshape(1, 1, 4, 4)
    camera_from_world[..., 0, 3] = -1.0
    point = _unproject(depth, intrinsics, camera_from_world)
    assert torch.allclose(point[0, 0, 0, 0], torch.tensor([1.0, 0.0, 1.0]))


def test_vggt_dependency_error_is_actionable() -> None:
    try:
        import vggt  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="official package"):
            GeometryBeliefHead(backend="vggt")
