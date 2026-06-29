from pathlib import Path

import numpy as np
import torch

from jepa4d.benchmarks.geometry.tum_rgbd import TUMSample, depth_metrics, point_metrics, pose_metrics


def test_aligned_depth_and_point_metrics_are_exact_for_scaled_prediction() -> None:
    target = torch.linspace(0.5, 2.0, 256).reshape(16, 16)
    predicted = target * 2.5
    metrics, scale, _ = depth_metrics(predicted, target)
    assert np.isclose(scale, 0.4)
    assert metrics["abs_rel"] < 1e-6
    intrinsics = torch.tensor([[16.0, 0.0, 7.5], [0.0, 16.0, 7.5], [0.0, 0.0, 1.0]])
    points = point_metrics(predicted, target, intrinsics)
    assert points["point_error_mean_m_aligned"] < 1e-6
    assert points["point_fscore_5cm_aligned"] == 1.0


def test_pose_metrics_validate_camera_from_world_convention() -> None:
    samples = []
    extrinsics = []
    for index in range(4):
        center = np.array([float(index), 0.1 * index, 0.0])
        samples.append(
            TUMSample(
                sample_id=str(index),
                timestamp=float(index),
                rgb_path=Path("rgb.png"),
                depth_path=Path("depth.png"),
                translation=center,
                quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            )
        )
        camera_from_world = torch.eye(4, dtype=torch.float64)
        camera_from_world[:3, 3] = -torch.from_numpy(center)
        extrinsics.append(camera_from_world)
    metrics = pose_metrics(torch.stack(extrinsics), samples)
    assert metrics["pose_ate_rmse_m_sim3"] < 1e-8
    assert metrics["pose_rotation_mean_deg_sim3"] < 1e-8
