# JEPA-4D API reference

## Phase 4 persistent-memory API

```python
from jepa4d.memory.memory_update import FourDMemoryCore
from jepa4d.memory.persistence import MemoryPersistence

persistence = MemoryPersistence("outputs/memory.db")
memory = FourDMemoryCore()
result = memory.update(
    geometry,
    object_slots,
    robot_state,
    timestamp=12.5,
    persistence=persistence,
)
snapshot = memory.snapshot()
loaded = FourDMemoryCore.load(persistence)
replayed = FourDMemoryCore.replay(persistence)
assert loaded.snapshot().to_serializable() == replayed.snapshot().to_serializable()
```

Updates require monotonic timestamps and atomically persist current records, event-log entries, and snapshots. The
returned `MemoryUpdateResult` reports revision, inserted/updated objects, local objects, episodic events, and persistence
records. `LODPolicy.compress(snapshot, task_context)` returns a bounded copy and never mutates live memory.

Planner-facing access remains through `WorldModelQueryAPI`, including `find_object`, `get_local_context`,
`get_observation_history`, `get_uncertainty`, `get_affordances`, route/region methods, verification, and task state.

## Phase 3 quick start

```python
from jepa4d.data.rgb_input import load_rgb_input
from jepa4d.models.geometry_belief import GeometryBeliefHead
from jepa4d.models.object_slot_grounder import ObjectSlotGrounder
from jepa4d.models.vjepa21_adapter import VJEPA21FeatureExtractor

batch = load_rgb_input(["view0.jpg", "view1.jpg"])
tokens = VJEPA21FeatureExtractor(mock=True)(batch)
geometry = GeometryBeliefHead()(batch)
result = ObjectSlotGrounder()(batch, ["mug", "table"], tokens=tokens, geometry=geometry)
for slot in result.slots:
    print(slot.object_id, slot.category, slot.pose_map, slot.observation_refs)
```

Use `detector_backend="grounding_dino"` for the real teacher and `mask_backend="sam2"` for optional prompted masks.
Teacher models are loaded lazily. `box` masks are detector baselines, not segmentations. `result.save_json()` records
summaries and `result.save_masks()` stores lossless masks in NPZ.

The end-to-end CLI is:

```bash
python -m jepa4d.cli.build_memory \
  --images view0.jpg --images view1.jpg \
  --query mug --query table \
  --detector-backend grounding_dino --mask-backend box \
  --output outputs/object_memory
python -m jepa4d.cli.query_memory --db outputs/object_memory/memory.db --query mug
```

Outputs include object JSON, masks NPZ, SQLite memory, scene graph JSON, interactive HTML, and a Markdown experiment
record. Add `--wandb` only when `WANDB_API_KEY` is supplied through the environment.

## 1. Stability levels

- **Stable Phase 1:** RGB contracts, token bundle, mock extraction, local V-JEPA 2.1 extraction, feature artifacts.
- **Stable Phase 2:** geometry belief contract, mock/VGGT adapter boundary, NPZ/PLY export, reconstruction CLI.
- **Stable Phase 3 substrate:** object observations/slots, mock/teacher boundary, artifact formats, and grounding CLI.
- **Stable Phase 4 substrate:** monotonic updates, active/global memory, event log, snapshots, reload/replay, and LOD.
- **Preview:** HTTP mutation schemas, frame transforms, identity repair, and production persistence migrations.
- **Reserved:** calibrated object permanence, latent dynamics, planner execution, and dataset evaluation CLIs.

Public APIs are typed and preserve view/time identity. Tensor shapes in this document are part of the contract.

## 2. RGB input

### `from_view_sequences`

```python
from jepa4d.data.rgb_input import from_view_sequences

single = from_view_sequences([[image]])
multiview = from_view_sequences([[left], [right], [wrist]])
video = from_view_sequences([[frame0, frame1, frame2, frame3]])
multiview_video = from_view_sequences([
    [left_t0, left_t1],
    [right_t0, right_t1],
])
```

Inputs may be paths, PIL images, HWC NumPy arrays, or CHW/HWC tensors. Values are converted to float RGB in `[0,1]`.
Every view must have the same number of timesteps and spatial dimensions in one sample.

### `load_rgb_input`

```python
batch = load_rgb_input(["view0.jpg", "view1.jpg"])
clip = load_rgb_input(["clip.mp4"], max_frames=16, stride=2)
```

A single recognized video suffix is decoded as video. Repeated image paths form a multi-view set.

### `collate_rgb_inputs`

Pads variable view/time samples to the batch maximum and creates `[B,V,T]` validity. It does not resize spatially; all
samples passed to one call must share dimensions.

### `RGBInputBatch`

Required fields: images, timestamps, camera IDs, and mode. Optional intrinsics are `[B,V,3,3]`; optional extrinsics are
`[B,V,4,4]`. Calibration currently applies to all timesteps for a view. `to(device)` moves tensor fields while retaining
IDs and references. `to_serializable()` returns metadata rather than embedding large tensors by default.

## 3. View-set identity

```python
from jepa4d.models.viewset_tokenizer import ViewSetTokenizer

encoding = ViewSetTokenizer(embed_dim=128)(batch)
```

`encoding.identity_tokens` is `[B,V,T,D]`; `images` and `valid_mask` remain unchanged. Identity tokens combine mode,
view index, and continuous time. The module supports up to `max_views`, default 32.

## 4. V-JEPA 2.1 features

### Mock

```python
extractor = VJEPA21FeatureExtractor(mock=True, mock_embed_dim=64)
bundle = extractor(batch)
```

### Real HF-compatible checkpoint

```python
extractor = VJEPA21FeatureExtractor(
    model_name="vjepa2_1_vit_base_384",
    checkpoint="checkpoints/vjepa2.1-vitb-fpc64-384",
    implementation_path="checkpoints/vjepa21_hf_impl",
    frozen=True,
    device="cpu",
)
bundle = extractor(batch)
```

### Real native checkpoint

Pass an official `.pt` checkpoint and `backend="native"`. Distilled ViT-B/L models load `ema_encoder`; larger models load
`target_encoder`.

### `JEPATokenBundle`

```text
dense_tokens  [B,V,T',576,C]
global_tokens [B,V,T',C]
layer_tokens  dict[layer -> B,V,T',576,C]
valid_mask    [B,V,T']
patch_grid    (24,24)
```

`save(path)` writes a PyTorch artifact. `write_metadata(path)` records shapes, model configuration, and runtime.

### Feature CLI

```bash
python -m jepa4d.cli.encode \
  --input view0.jpg --input view1.jpg \
  --output outputs/features.zarr \
  --model vjepa2_1_vit_base_384 \
  --checkpoint checkpoints/vjepa2.1-vitb-fpc64-384 \
  --device cpu \
  --wandb
```

Output may be `.pt` or `.zarr`. The containing directory also receives `metadata.json`, `report.html`, and
`EXPERIMENT.md`.

## 5. Geometry belief

### Deterministic mock

```python
from jepa4d.models.geometry_belief import GeometryBeliefHead

head = GeometryBeliefHead(backend="mock", output_size=112, query_grid_size=8)
belief = head(batch)
```

This backend validates integration only. `metadata["synthetic_geometry"]` is `True`.

### Official VGGT

Install the official package and obtain its checkpoint:

```bash
pip install --no-deps 'git+https://github.com/facebookresearch/vggt.git'
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "facebook/VGGT-1B",
    local_dir="checkpoints/VGGT-1B",
    allow_patterns=["config.json", "model.safetensors", "README.md"],
)
PY
```

Then:

```python
head = GeometryBeliefHead(
    backend="vggt",
    model_id="checkpoints/VGGT-1B",
    device="cuda",
)
belief = head(batch)
```

The checkpoint is approximately 5.03 GB. CPU inference is supported as a smoke test but is not the target runtime.

### `GeometryBelief`

```text
camera_extrinsics        [B,V,T,4,4] or None
camera_intrinsics        [B,V,T,3,3] or None
depth_mean/logvar        [B,V,T,Hg,Wg] or None
pointmap_mean/logvar     [B,V,T,Hg,Wg,3] or None
tracks_2d               [B,V*T,N,2] or None
tracks_3d               [B,V*T,N,3] or None
scale_confidence         [B]
pose_confidence          [B]
reconstruction_confidence [B]
```

All confidence values are bounded `[0,1]`. The current mapping is a heuristic belief score pending benchmark
calibration. Use `to_serializable()` for metadata and `save_npz(path)` for complete arrays.

### Geometry export

```python
from jepa4d.models.geometry_export import export_geometry_npz, export_pointcloud_ply

export_geometry_npz(belief, "geometry.npz")
export_pointcloud_ply(
    belief,
    batch,
    "pointcloud.ply",
    max_points=100_000,
    max_logvar=1.0,
)
```

PLY contains world-frame XYZ and RGB. Non-finite points are removed. `max_logvar` optionally excludes uncertain points.

### Reconstruction CLI

```bash
python -m jepa4d.cli.reconstruct \
  --images view0.jpg --images view1.jpg --images view2.jpg \
  --backend vggt \
  --model-id checkpoints/VGGT-1B \
  --device cuda \
  --output outputs/reconstruction \
  --wandb
```

For contract-only testing, use `--backend mock --output-size 112`. Outputs are `geometry_belief.npz`, `pointcloud.ply`,
`metadata.json`, `report.html`, and `EXPERIMENT.md`.

## 6. Uncertainty and verification

```python
if belief.scale_confidence[0] < 0.5:
    action = query_api.suggest_verification_action(entity_id)
```

`known_scale_prior=True` means the caller has supplied a legitimate metric prior such as calibrated stereo baseline or a
known robot/object dimension. It must not be enabled merely to increase confidence. Known intrinsics and extrinsics should
be inserted into `RGBInputBatch` instead.

## 7. Memory and queries

```python
api = WorldModelQueryAPI()
api.find_object("red mug", region="kitchen")
api.get_region_summary("kitchen")
api.get_route("hall", "kitchen")
api.get_local_context(radius_m=5.0, frame="base_link")
api.get_observation_history("mug-1")
api.verify_condition("mug on table")
api.get_affordances("mug-1")
api.get_uncertainty("mug-1")
api.suggest_verification_action("mug-1")
api.mark_task_state("place-mug", "verified", evidence)
```

Current matching is simple text containment and routes use graph breadth-first search. These are interface implementations,
not final semantic planners.

## 8. HTTP API

Start:

```bash
uvicorn jepa4d.server.app:app --host 0.0.0.0 --port 8000
```

Routes:

- `GET /health`
- `POST /memory/update`
- `POST /query/find_object`
- `GET /query/region/{region_id}`
- `POST /query/verify`
- `POST /planner/plan`
- `POST /planner/replan`

Planner responses are explicitly marked mock until Phase 5.

## 9. Observability API

`ExperimentLogger` can be disabled, offline, or online. Feature and geometry log methods accept typed contracts.
`log_training_step` reserves detailed scalar names for future optimization loops. API keys must be supplied through
`WANDB_API_KEY`; they must never be added to YAML, Markdown, shell scripts, or source code.

## 10. Errors callers should handle

- invalid `[B,V,T]` mode/shape combinations;
- inconsistent view sequence lengths or spatial dimensions;
- missing real checkpoints;
- missing optional VGGT installation;
- unavailable point map during PLY export;
- confidence outside `[0,1]`;
- CUDA initialization or out-of-memory failures;
- video decoding failures;
- W&B network failures when online logging is requested.
