"""Open-vocabulary detections, masks, and persistent object-slot association."""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from jepa4d.data.schemas import JEPATokenBundle, RGBInputBatch
from jepa4d.models.geometry_belief import GeometryBelief

DetectorBackend = Literal["mock", "grounding_dino"]
MaskBackend = Literal["box", "sam2"]


@dataclass(frozen=True, slots=True)
class AssociationConfig:
    appearance_weight: float = 0.65
    iou_weight: float = 0.20
    geometry_weight: float = 0.15
    threshold: float = 0.55
    max_time_gap: int = 8
    geometry_distance_scale_m: float = 1.0

    def __post_init__(self) -> None:
        weights = (self.appearance_weight, self.iou_weight, self.geometry_weight)
        if any(value < 0 for value in weights) or sum(weights) <= 0:
            raise ValueError("association weights must be non-negative with a positive sum")
        if not 0 <= self.threshold <= 1:
            raise ValueError("association threshold must be within [0,1]")
        if self.max_time_gap < 0 or self.geometry_distance_scale_m <= 0:
            raise ValueError("association gap and geometry scale must be positive")

    @property
    def normalized_weights(self) -> tuple[float, float, float]:
        total = self.appearance_weight + self.iou_weight + self.geometry_weight
        return (
            self.appearance_weight / total,
            self.iou_weight / total,
            self.geometry_weight / total,
        )


@dataclass(slots=True)
class ObjectObservation:
    """One object hypothesis tied to an original view and timestep."""

    observation_id: str
    batch_index: int
    view_index: int
    time_index: int
    camera_id: str
    category: str
    score: float
    bbox_2d: list[float]
    mask: np.ndarray
    visual_embedding: np.ndarray
    pose_map: list[float] | None = None

    def summary(self) -> dict[str, Any]:
        value = asdict(self)
        value["mask"] = {"shape": list(self.mask.shape), "area": int(self.mask.sum())}
        value["visual_embedding"] = {
            "shape": list(self.visual_embedding.shape),
            "norm": float(np.linalg.norm(self.visual_embedding)),
        }
        return value


@dataclass(slots=True)
class ObjectSlot:
    object_id: str
    category: str = "unknown"
    description: str = ""
    mask: np.ndarray | None = None
    bbox_2d: list[float] | None = None
    bbox_3d: list[float] | None = None
    pose_map: list[float] | None = None
    pose_robot: list[float] | None = None
    visual_embedding: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float32))
    language_embedding: np.ndarray | None = None
    affordances: dict[str, float] = field(default_factory=dict)
    states: dict[str, float] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)
    last_seen_time: float = 0.0
    observation_refs: list[str] = field(default_factory=list)
    observations: list[ObjectObservation] = field(default_factory=list)

    def to_serializable(self, include_mask: bool = False, include_embedding: bool = True) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "category": self.category,
            "description": self.description,
            "mask": None
            if self.mask is None
            else (
                self.mask.tolist() if include_mask else {"shape": list(self.mask.shape), "area": int(self.mask.sum())}
            ),
            "bbox_2d": self.bbox_2d,
            "bbox_3d": self.bbox_3d,
            "pose_map": self.pose_map,
            "pose_robot": self.pose_robot,
            "visual_embedding": self.visual_embedding.tolist()
            if include_embedding
            else {"shape": list(self.visual_embedding.shape)},
            "language_embedding": None if self.language_embedding is None else self.language_embedding.tolist(),
            "affordances": self.affordances,
            "states": self.states,
            "confidence": self.confidence,
            "last_seen_time": self.last_seen_time,
            "observation_refs": self.observation_refs,
            "observations": [observation.summary() for observation in self.observations],
        }


@dataclass(slots=True)
class ObjectGroundingResult:
    slots: list[ObjectSlot]
    observations: list[ObjectObservation]
    queries: list[str]
    metadata: dict[str, Any]

    def to_serializable(self) -> dict[str, Any]:
        return {
            "queries": self.queries,
            "slots": [slot.to_serializable() for slot in self.slots],
            "observations": [observation.summary() for observation in self.observations],
            "metadata": self.metadata,
        }

    def save_json(self, path: str | Path) -> Path:
        import json

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_serializable(), indent=2) + "\n")
        return target

    def save_masks(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        values = {value.observation_id: value.mask.astype(np.uint8) for value in self.observations}
        np.savez_compressed(target, **values)  # type: ignore[arg-type]
        return target


def _normalize_embedding(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return value / max(float(np.linalg.norm(value)), 1e-8)


def _bbox_iou(first: list[float], second: list[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(area_first + area_second - intersection, 1e-8)


class ObjectSlotGrounder(nn.Module):
    """Ground text queries and associate detections across views and time.

    `mock` produces deterministic image-conditioned boxes and masks. The real
    detector uses the Transformers Grounding DINO implementation. SAM2 mask
    refinement is optional; `box` masks remain a useful detector-only baseline.
    """

    def __init__(
        self,
        *,
        detector_backend: str = "mock",
        mask_backend: str = "box",
        detector_model_id: str = "IDEA-Research/grounding-dino-tiny",
        sam2_model_id: str = "facebook/sam2-hiera-tiny",
        device: str | torch.device = "cpu",
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        association_threshold: float = 0.55,
        appearance_weight: float = 0.65,
        iou_weight: float = 0.20,
        geometry_weight: float = 0.15,
        max_time_gap: int = 8,
        geometry_distance_scale_m: float = 1.0,
    ) -> None:
        super().__init__()
        if detector_backend not in {"mock", "grounding_dino"}:
            raise ValueError(f"unknown detector backend: {detector_backend}")
        if mask_backend not in {"box", "sam2"}:
            raise ValueError(f"unknown mask backend: {mask_backend}")
        self.detector_backend = detector_backend
        self.mask_backend = mask_backend
        self.detector_model_id = detector_model_id
        self.sam2_model_id = sam2_model_id
        self.device_name = str(device)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.association = AssociationConfig(
            appearance_weight=appearance_weight,
            iou_weight=iou_weight,
            geometry_weight=geometry_weight,
            threshold=association_threshold,
            max_time_gap=max_time_gap,
            geometry_distance_scale_m=geometry_distance_scale_m,
        )
        self.detector: nn.Module | None = None
        self.processor: Any = None
        self.mask_predictor: Any = None
        self._load_seconds = 0.0
        self._load_teachers()

    def _load_teachers(self) -> None:
        started = time.perf_counter()
        if self.detector_backend == "grounding_dino":
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(self.detector_model_id)
            self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(self.detector_model_id)
            self.detector.to(self.device_name).eval()
            for parameter in self.detector.parameters():
                parameter.requires_grad_(False)
        if self.mask_backend == "sam2":
            try:
                from sam2.sam2_image_predictor import SAM2ImagePredictor
            except ImportError as error:
                raise ImportError(
                    "SAM2 mask backend requested but the official package is unavailable. "
                    "Install from https://github.com/facebookresearch/sam2."
                ) from error
            self.mask_predictor = SAM2ImagePredictor.from_pretrained(self.sam2_model_id)
        self._load_seconds = time.perf_counter() - started

    def forward(
        self,
        batch: RGBInputBatch,
        queries: list[str],
        *,
        tokens: JEPATokenBundle | None = None,
        geometry: GeometryBelief | None = None,
    ) -> ObjectGroundingResult:
        if batch.images.shape[0] != 1:
            raise ValueError("ObjectSlotGrounder currently expects B=1; collated batches must be processed per sample")
        clean_queries = sorted({value.strip().lower() for value in queries if value.strip()})
        if not clean_queries:
            raise ValueError("at least one non-empty object query is required")
        started = time.perf_counter()
        observations = (
            self._detect_mock(batch, clean_queries, tokens, geometry)
            if self.detector_backend == "mock"
            else self._detect_grounding_dino(batch, clean_queries, tokens, geometry)
        )
        slots = self.associate_observations(observations, batch)
        return ObjectGroundingResult(
            slots=slots,
            observations=observations,
            queries=clean_queries,
            metadata={
                "detector_backend": self.detector_backend,
                "mask_backend": self.mask_backend,
                "detector_model_id": self.detector_model_id
                if self.detector_backend != "mock"
                else "deterministic_mock",
                "sam2_model_id": self.sam2_model_id if self.mask_backend == "sam2" else None,
                "runtime_seconds": time.perf_counter() - started,
                "model_load_seconds": self._load_seconds,
                "input_mode": batch.mode,
                "input_shape": list(batch.images.shape),
                "query_count": len(clean_queries),
                "observation_count": len(observations),
                "slot_count": len(slots),
                "association": asdict(self.association),
                "uses_jepa_tokens": tokens is not None,
                "uses_geometry": geometry is not None,
                "mock_outputs_are_not_accuracy_predictions": self.detector_backend == "mock",
            },
        )

    def _detect_mock(
        self,
        batch: RGBInputBatch,
        queries: list[str],
        tokens: JEPATokenBundle | None,
        geometry: GeometryBelief | None,
    ) -> list[ObjectObservation]:
        observations: list[ObjectObservation] = []
        _, views, steps, _, height, width = batch.images.shape
        for view in range(views):
            for time_index in range(steps):
                image = batch.images[0, view, time_index]
                for query_index, query in enumerate(queries):
                    digest = hashlib.sha256(query.encode()).digest()
                    center_x = (0.25 + 0.5 * digest[0] / 255 + 0.025 * view) % 0.8 + 0.1
                    center_y = (0.25 + 0.5 * digest[1] / 255 + 0.015 * time_index) % 0.8 + 0.1
                    box_width = 0.18 + 0.08 * digest[2] / 255
                    box_height = 0.18 + 0.08 * digest[3] / 255
                    bbox = [
                        max(0.0, (center_x - box_width / 2) * width),
                        max(0.0, (center_y - box_height / 2) * height),
                        min(float(width), (center_x + box_width / 2) * width),
                        min(float(height), (center_y + box_height / 2) * height),
                    ]
                    mask = self._box_mask(bbox, height, width)
                    embedding = self.extract_visual_embedding(image, bbox, query, tokens, view, time_index, mask=mask)
                    pose = self._geometry_centroid(mask, geometry, view, time_index)
                    observation_id = f"b0-v{view}-t{time_index}-q{query_index}"
                    observations.append(
                        ObjectObservation(
                            observation_id=observation_id,
                            batch_index=0,
                            view_index=view,
                            time_index=time_index,
                            camera_id=batch.camera_ids[0][view],
                            category=query,
                            score=0.55 + 0.35 * digest[4] / 255,
                            bbox_2d=bbox,
                            mask=mask,
                            visual_embedding=embedding,
                            pose_map=pose,
                        )
                    )
        return observations

    @staticmethod
    def _box_mask(bbox: list[float], height: int, width: int) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        mask = np.zeros((height, width), dtype=bool)
        mask[max(0, int(y1)) : min(height, int(np.ceil(y2))), max(0, int(x1)) : min(width, int(np.ceil(x2)))] = True
        return mask

    def extract_visual_embedding(
        self,
        image: torch.Tensor,
        bbox: list[float],
        query: str,
        tokens: JEPATokenBundle | None,
        view: int,
        time_index: int,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        if tokens is not None:
            token_time = min(
                time_index // 2 if tokens.modality == "video" else time_index, tokens.dense_tokens.shape[2] - 1
            )
            grid_h, grid_w = tokens.patch_grid
            height, width = image.shape[-2:]
            x1 = max(0, min(grid_w - 1, int(bbox[0] / width * grid_w)))
            x2 = max(x1 + 1, min(grid_w, int(np.ceil(bbox[2] / width * grid_w))))
            y1 = max(0, min(grid_h - 1, int(bbox[1] / height * grid_h)))
            y2 = max(y1 + 1, min(grid_h, int(np.ceil(bbox[3] / height * grid_h))))
            grid = tokens.dense_tokens[0, view, token_time].reshape(grid_h, grid_w, -1)
            if mask is not None:
                mask_tensor = torch.from_numpy(mask).float()[None, None]
                grid_mask = F.interpolate(mask_tensor, size=(grid_h, grid_w), mode="nearest")[0, 0].bool()
                if grid_mask.any():
                    return _normalize_embedding(grid[grid_mask].float().mean(dim=0).detach().cpu().numpy())
            return _normalize_embedding(grid[y1:y2, x1:x2].float().mean(dim=(0, 1)).detach().cpu().numpy())
        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        crop = image[:, max(0, y1) : max(y1 + 1, y2), max(0, x1) : max(x1 + 1, x2)].float()
        statistics = torch.cat((crop.mean(dim=(1, 2)), crop.std(dim=(1, 2)))).cpu().numpy()
        digest = np.frombuffer(hashlib.sha256(query.encode()).digest()[:26], dtype=np.uint8).astype(np.float32) / 255
        return _normalize_embedding(np.concatenate((statistics, digest)))

    def _geometry_centroid(
        self, mask: np.ndarray, geometry: GeometryBelief | None, view: int, time_index: int
    ) -> list[float] | None:
        if geometry is None or geometry.pointmap_mean is None:
            return None
        points = geometry.pointmap_mean[0, view, time_index]
        mask_tensor = torch.from_numpy(mask).float().to(points.device)[None, None]
        resized = F.interpolate(mask_tensor, size=points.shape[:2], mode="nearest")[0, 0].bool()
        selected = points[resized & torch.isfinite(points).all(dim=-1)]
        return None if selected.numel() == 0 else selected.float().mean(dim=0).detach().cpu().tolist()

    def _detect_grounding_dino(
        self,
        batch: RGBInputBatch,
        queries: list[str],
        tokens: JEPATokenBundle | None,
        geometry: GeometryBelief | None,
    ) -> list[ObjectObservation]:
        assert self.detector is not None and self.processor is not None
        observations: list[ObjectObservation] = []
        _, views, steps, _, height, width = batch.images.shape
        prompt = ". ".join(queries) + "."
        for view in range(views):
            for time_index in range(steps):
                tensor = batch.images[0, view, time_index]
                image_np = (tensor.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                image = Image.fromarray(image_np)
                inputs = self.processor(images=image, text=prompt, return_tensors="pt")
                inputs = {key: value.to(self.device_name) for key, value in inputs.items()}
                with torch.inference_mode():
                    outputs = self.detector(**inputs)
                results = self.processor.post_process_grounded_object_detection(
                    outputs,
                    inputs["input_ids"],
                    threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    target_sizes=[(height, width)],
                )[0]
                boxes = results["boxes"].detach().cpu().tolist()
                scores = results["scores"].detach().cpu().tolist()
                labels = results["text_labels"] if "text_labels" in results else results.get("labels", [])
                labels = [str(value) for value in labels]
                masks = self._segment_masks(image_np, boxes)
                for detection_index, (bbox, score, label, mask) in enumerate(
                    zip(boxes, scores, labels, masks, strict=True)
                ):
                    category = self._canonical_category(label, queries)
                    embedding = self.extract_visual_embedding(
                        tensor, bbox, category, tokens, view, time_index, mask=mask
                    )
                    pose = self._geometry_centroid(mask, geometry, view, time_index)
                    observations.append(
                        ObjectObservation(
                            observation_id=f"b0-v{view}-t{time_index}-d{detection_index}",
                            batch_index=0,
                            view_index=view,
                            time_index=time_index,
                            camera_id=batch.camera_ids[0][view],
                            category=category,
                            score=float(score),
                            bbox_2d=[float(value) for value in bbox],
                            mask=mask,
                            visual_embedding=embedding,
                            pose_map=pose,
                        )
                    )
        return observations

    @staticmethod
    def _canonical_category(label: str, queries: list[str]) -> str:
        lowered = label.lower()
        matches = [query for query in queries if query in lowered or lowered in query]
        return max(matches, key=len) if matches else lowered.strip(" .")

    def _segment_masks(self, image: np.ndarray, boxes: list[list[float]]) -> list[np.ndarray]:
        if self.mask_backend == "box":
            return [self._box_mask(box, image.shape[0], image.shape[1]) for box in boxes]
        assert self.mask_predictor is not None
        self.mask_predictor.set_image(image)
        masks = []
        for box in boxes:
            predicted, scores, _ = self.mask_predictor.predict(box=np.asarray(box), multimask_output=True)
            masks.append(np.asarray(predicted[int(np.argmax(scores))], dtype=bool))
        return masks

    def associate_observations(self, observations: list[ObjectObservation], batch: RGBInputBatch) -> list[ObjectSlot]:
        """Associate externally supplied observations with frame-wise exclusivity."""
        clusters: list[list[ObjectObservation]] = []
        grouped: dict[tuple[str, int, int], list[ObjectObservation]] = {}
        for observation in observations:
            grouped.setdefault((observation.category, observation.time_index, observation.view_index), []).append(
                observation
            )
        for (category, time_index, view_index), frame_observations in sorted(grouped.items()):
            candidates: list[tuple[float, int, int]] = []
            for cluster_index, cluster in enumerate(clusters):
                representative = max(cluster, key=lambda value: (value.time_index, value.view_index))
                time_gap = time_index - representative.time_index
                if representative.category != category or time_gap < 0 or time_gap > self.association.max_time_gap:
                    continue
                if any(value.time_index == time_index and value.view_index == view_index for value in cluster):
                    continue
                for observation_index, observation in enumerate(frame_observations):
                    candidates.append(
                        (
                            self._association_score(cluster, observation),
                            cluster_index,
                            observation_index,
                        )
                    )
            used_clusters: set[int] = set()
            used_observations: set[int] = set()
            for score, cluster_index, observation_index in sorted(candidates, reverse=True):
                if score < self.association.threshold:
                    break
                if cluster_index in used_clusters or observation_index in used_observations:
                    continue
                clusters[cluster_index].append(frame_observations[observation_index])
                used_clusters.add(cluster_index)
                used_observations.add(observation_index)
            for observation_index, observation in enumerate(frame_observations):
                if observation_index not in used_observations:
                    clusters.append([observation])
        slots = []
        for cluster_index, cluster in enumerate(clusters):
            category = cluster[0].category
            poses = [np.asarray(value.pose_map) for value in cluster if value.pose_map is not None]
            pose = None if not poses else np.mean(poses, axis=0).tolist()
            embedding = _normalize_embedding(np.mean([value.visual_embedding for value in cluster], axis=0))
            stable_key = f"{category}:{cluster_index}:" + (
                "none" if pose is None else ":".join(f"{x:.2f}" for x in pose)
            )
            object_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"jepa4d:{stable_key}"))
            latest = max(cluster, key=lambda value: (value.time_index, value.view_index))
            score = float(np.mean([value.score for value in cluster]))
            slots.append(
                ObjectSlot(
                    object_id=object_id,
                    category=category,
                    description=f"grounded {category}",
                    mask=latest.mask,
                    bbox_2d=latest.bbox_2d,
                    pose_map=pose,
                    visual_embedding=embedding,
                    affordances=self._default_affordances(category),
                    states={"visible": 1.0},
                    confidence={
                        "detection": score,
                        "association": min(1.0, len(cluster) / max(1, batch.images.shape[1] * batch.images.shape[2])),
                        "geometry": 0.0 if pose is None else 0.5,
                        "overall": 0.7 * score + 0.3 * min(1.0, len(cluster) / 2),
                    },
                    last_seen_time=float(batch.timestamps[0, latest.view_index, latest.time_index]),
                    observation_refs=[value.observation_id for value in cluster],
                    observations=cluster,
                )
            )
        return slots

    def _association_score(self, cluster: list[ObjectObservation], observation: ObjectObservation) -> float:
        representative = max(cluster, key=lambda value: (value.time_index, value.view_index))
        cluster_embedding = _normalize_embedding(np.mean([value.visual_embedding for value in cluster[-4:]], axis=0))
        appearance = float(np.clip(cluster_embedding @ observation.visual_embedding, -1.0, 1.0))
        appearance = (appearance + 1.0) / 2.0
        overlap = _bbox_iou(representative.bbox_2d, observation.bbox_2d)
        geometry_score = 0.0
        if representative.pose_map is not None and observation.pose_map is not None:
            distance = np.linalg.norm(np.asarray(representative.pose_map) - np.asarray(observation.pose_map))
            geometry_score = float(np.exp(-distance / self.association.geometry_distance_scale_m))
        appearance_weight, iou_weight, geometry_weight = self.association.normalized_weights
        return appearance_weight * appearance + iou_weight * overlap + geometry_weight * geometry_score

    @staticmethod
    def _default_affordances(category: str) -> dict[str, float]:
        category = category.lower()
        return {
            "graspable": 0.8 if any(word in category for word in ("mug", "cup", "bottle", "box")) else 0.3,
            "openable": 0.8 if any(word in category for word in ("drawer", "door", "cabinet", "box")) else 0.05,
            "support_surface": 0.8 if any(word in category for word in ("table", "counter", "shelf")) else 0.1,
        }
