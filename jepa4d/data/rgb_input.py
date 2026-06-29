"""Load and collate RGB images and videos into the canonical contract."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from jepa4d.data.schemas import InputMode, RGBInputBatch


def _as_chw_float(image: str | Path | Image.Image | np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(image, (str, Path)):
        value = np.asarray(Image.open(image).convert("RGB"), dtype=np.uint8).copy()
    elif isinstance(image, Image.Image):
        value = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    elif isinstance(image, np.ndarray):
        value = image
    else:
        tensor = image.detach()
        if tensor.ndim != 3:
            raise ValueError("each image must be a rank-3 HWC or CHW tensor")
        if tensor.shape[0] == 3:
            result = tensor.float()
        elif tensor.shape[-1] == 3:
            result = tensor.permute(2, 0, 1).float()
        else:
            raise ValueError("cannot identify RGB channel dimension")
        return result / 255.0 if result.max() > 1.0 else result
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError("numpy/PIL images must have shape [H,W,3]")
    return torch.from_numpy(np.ascontiguousarray(value)).permute(2, 0, 1).float() / 255.0


def infer_mode(views: int, steps: int) -> InputMode:
    if views == 1 and steps == 1:
        return "single_image"
    if views > 1 and steps == 1:
        return "multi_view"
    if views == 1:
        return "video"
    return "multiview_video"


def from_view_sequences(
    views: Sequence[Sequence[str | Path | Image.Image | np.ndarray | torch.Tensor]],
    *,
    timestamps: torch.Tensor | None = None,
    camera_ids: Sequence[str] | None = None,
) -> RGBInputBatch:
    """Create a single-item batch from V view sequences, each containing T frames."""
    if not views or not views[0]:
        raise ValueError("at least one view and one frame are required")
    steps = len(views[0])
    if any(len(view) != steps for view in views):
        raise ValueError("all views must have the same number of timesteps")
    tensors = [[_as_chw_float(frame) for frame in view] for view in views]
    shape = tensors[0][0].shape
    if any(frame.shape != shape for view in tensors for frame in view):
        raise ValueError("all images must share a spatial resolution")
    images = torch.stack([torch.stack(view, dim=0) for view in tensors], dim=0).unsqueeze(0)
    view_count = len(views)
    ts = timestamps if timestamps is not None else torch.arange(steps, dtype=torch.float64).repeat(view_count, 1)
    if tuple(ts.shape) != (view_count, steps):
        raise ValueError("timestamps must have shape [V,T]")
    ids = list(camera_ids) if camera_ids is not None else [f"camera_{index}" for index in range(view_count)]
    refs = [[[str(frame) if isinstance(frame, (str, Path)) else "in_memory" for frame in view] for view in views]]
    return RGBInputBatch(
        images=images,
        timestamps=ts.unsqueeze(0),
        camera_ids=[ids],
        mode=infer_mode(view_count, steps),
        source_refs=refs,
    )


def _load_video(path: Path, max_frames: int | None = None, stride: int = 1) -> list[np.ndarray]:
    import imageio.v3 as iio

    frames = []
    for index, frame in enumerate(iio.imiter(path, plugin="FFMPEG")):
        if index % stride:
            continue
        frames.append(np.asarray(frame)[..., :3])
        if max_frames is not None and len(frames) >= max_frames:
            break
    if not frames:
        raise ValueError(f"video contains no readable frames: {path}")
    return frames


def load_rgb_input(
    inputs: Sequence[str | Path], *, as_video: bool | None = None, max_frames: int | None = 16, stride: int = 1
) -> RGBInputBatch:
    """Load one image, a multi-view image set, or one short video."""
    paths = [Path(value) for value in inputs]
    if not paths:
        raise ValueError("no input paths provided")
    video_suffixes = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    is_video = as_video if as_video is not None else len(paths) == 1 and paths[0].suffix.lower() in video_suffixes
    if is_video:
        return from_view_sequences([_load_video(paths[0], max_frames=max_frames, stride=stride)])
    return from_view_sequences([[path] for path in paths])


def collate_rgb_inputs(samples: Iterable[RGBInputBatch]) -> RGBInputBatch:
    """Pad variable view/time counts while retaining a valid observation mask."""
    items = list(samples)
    if not items:
        raise ValueError("cannot collate an empty batch")
    if any(item.images.shape[0] != 1 for item in items):
        raise ValueError("collate expects individual samples with B=1")
    channels, height, width = items[0].images.shape[-3:]
    if any(item.images.shape[-3:] != (channels, height, width) for item in items):
        raise ValueError("all samples must have the same image resolution")
    max_views = max(item.images.shape[1] for item in items)
    max_steps = max(item.images.shape[2] for item in items)
    images = torch.zeros((len(items), max_views, max_steps, channels, height, width), dtype=items[0].images.dtype)
    timestamps = torch.zeros((len(items), max_views, max_steps), dtype=torch.float64)
    valid = torch.zeros((len(items), max_views, max_steps), dtype=torch.bool)
    camera_ids: list[list[str]] = []
    for batch_index, item in enumerate(items):
        views, steps = item.images.shape[1:3]
        images[batch_index, :views, :steps] = item.images[0]
        timestamps[batch_index, :views, :steps] = item.timestamps[0]
        valid[batch_index, :views, :steps] = item.valid_mask[0]
        camera_ids.append(item.camera_ids[0] + [f"padding_{i}" for i in range(max_views - views)])
    return RGBInputBatch(
        images=images,
        timestamps=timestamps,
        camera_ids=camera_ids,
        mode=infer_mode(max_views, max_steps),
        valid_mask=valid,
    )
