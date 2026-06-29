# GPU runtime and experiment observability

## Why the first Phase 3 run looked incomplete

The first object-grounding run initialized W&B after V-JEPA feature extraction, geometry inference, and teacher loading.
Consequently, W&B observed only the final grounding call. It had useful object counts, scores, a slot table, a mask overlay,
and artifacts, but it could not answer basic end-to-end questions: which stage dominated latency, whether real JEPA and
geometry ran, how feature distributions looked, how geometry confidence behaved, or what hardware was actually used.

That ordering has been corrected. W&B now starts immediately after RGB loading and before all heavyweight constructors
and forwards. The Phase 3 CLI logs an ordered four-stage semantic trace plus a six-stage latency trace.

## Complete Phase 3 logging contract

### Configuration

Every run records queries, detector/mask model IDs, geometry backend/model ID, V-JEPA checkpoint or mock status,
requested device, canonical input contract, input mode, views/timesteps, and tags for phase, stage, backend, and device.

### V-JEPA feature stage

- model and total feature time;
- frame throughput, token count, feature width, and temporal bins;
- mean, standard deviation, extrema, finite fraction, and global-token variance;
- token-value and L2-norm histograms;
- per-layer mean, standard deviation, norm, shape table, and scalar panels;
- dense-token PCA image;
- adjacent temporal cosine plot when the input has multiple temporal bins;
- original input image.

### Geometry stage

- runtime, view/time counts, and track count;
- scale, pose, and reconstruction confidence;
- depth mean, standard deviation, extrema, histogram, and depth image;
- log-variance mean/p95 and uncertainty image;
- finite point fraction and XYZ extent table.

Confidence is logged as model belief, not accuracy. Accuracy panels require ground truth and belong to benchmark runs.

### Object stage

- teacher load and grounding times;
- query, observation, and slot counts;
- mean observations per slot and track-length histogram;
- score mean/minimum and score histogram;
- mask/box area means and histograms;
- query-detection coverage and geometry-attached fraction;
- slot summary table;
- complete per-observation table with source view/time/camera;
- per-query detections, slots, and score table;
- detections-by-view/time chart;
- combined semantic-mask and scored-box overlay.

### Pipeline and hardware stage

- repeated scalar latency series, one point per stage;
- cumulative latency series;
- interactive stage latency bar chart and table;
- individually named timing scalars;
- total pipeline time;
- PyTorch CUDA build, resolved device, and CUDA availability;
- peak allocated/reserved GPU memory when CUDA is available;
- W&B automatic process, CPU, memory, disk, network, and GPU telemetry.

### Artifacts

The run versions object JSON, lossless masks, SQLite memory, and interactive HTML. Local outputs also include scene-graph
JSON and an experiment Markdown record. W&B is the comparison dashboard; local machine-readable artifacts remain the
reproducible source.

## Current A100 state

The host exposes an NVIDIA A100 80 GB PCIe at `0000:65:00.0`, but the device is currently unavailable:

- the original environment had PyTorch `2.12.1+cu130`, which is incompatible with driver 535.309.01;
- an isolated `.conda-gpu` environment now has official PyTorch `2.5.1+cu121` and torchvision `0.20.1+cu121`;
- CUDA 12.1 removes the version mismatch but still reports zero available devices;
- `nvidia-smi` reports `Unable to determine the device handle ... Unknown Error`;
- `lspci` reports revision `ff` and unknown header type `7f` for the A100.

The final two signals indicate that the PCI device is not responding to the host. A Python package change cannot repair
this. A host administrator must reset/rebind the GPU or reboot the node after confirming that no other workload owns it.
JEPA-4D must not label a CPU run as GPU merely because an A100 is physically listed.

## GPU health check

After the device is restored, run:

```bash
.conda-gpu/bin/python scripts/check_cuda.py
```

Success must report `cuda_available: true`, device name `NVIDIA A100 80GB PCIe`, and a nonzero device count. Then run a
small allocation before model loading:

```bash
.conda-gpu/bin/python - <<'PY'
import torch
x = torch.randn(4096, 4096, device="cuda")
y = x @ x
torch.cuda.synchronize()
print(torch.cuda.get_device_name(0), y.mean().item())
PY
```

## Complete real pipeline command

Once CUDA passes:

```bash
WANDB_API_KEY=... .conda-gpu/bin/python -m jepa4d.cli.build_memory \
  --images assets/architecture_vjepa2_1.jpg \
  --query person --query robot \
  --jepa-checkpoint checkpoints/vjepa2.1-vitb-fpc64-384 \
  --geometry-backend vggt --geometry-model-id checkpoints/VGGT-1B \
  --detector-backend grounding_dino --mask-backend box \
  --device cuda --wandb \
  --run-name phase3-full-real-pipeline-a100
```

The official pinned VGGT package is installed in `.conda-gpu`. Checkpoint and teacher revisions must remain unchanged
when comparing CPU and GPU latency.

## Dashboard reading order

1. Confirm config and `system/device`; reject mislabeled runs.
2. Read pipeline latency and cumulative latency to locate bottlenecks.
3. Check finite fractions and feature distributions for numerical faults.
4. Check geometry uncertainty before interpreting object poses.
5. Inspect query coverage, score/area distributions, and per-observation evidence.
6. Inspect overlays and the interactive HTML for qualitative failures.
7. Use benchmark runs—not inference confidence—to make accuracy claims.

## Remaining observability work

- dataset ground-truth panels for AP, IoU, pose error, IDF1/HOTA, and calibration;
- CUDA kernel/memory profiling with `torch.profiler` after GPU recovery;
- threshold sweeps logged as grouped runs rather than mixing them into one inference;
- multi-frame association matrices and identity-switch timelines;
- explicit error finalization so failed runs carry a structured failure stage;
- artifact checksums and checkpoint hashes in W&B summary as well as Markdown.
