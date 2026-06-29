# Real V-JEPA identity association ablation

## Question

Do frozen V-JEPA2.1 dense tokens improve persistent same-category object identity relative to RGB, IoU, and geometry
evidence, particularly through occlusion and re-entry?

## Code and model

- date: 2026-06-29 UTC;
- baseline commit before this work: `e14fcce`;
- model: local V-JEPA2.1 ViT-B 384 checkpoint;
- runtime: CPU because the A100 remained unavailable at PCI revision `ff`;
- association: frame-wise exclusive greedy matching;
- outputs committed only as code/docs; dataset and generated artifacts are ignored.

## Data

### Controlled fixture

Two same-category objects, ten frames, crossing motion, independent disappearance/re-entry, 17 observations.

### DAVIS 2017

- official page: <https://davischallenge.org/davis2017/code.html>;
- archive: `DAVIS-2017-trainval-480p.zip`;
- SHA-256: `e3d0b5b77c3d031b000a19e0e25e3e2cac65d183755601bc2cf066df1a2aa492`;
- sequence: `dogs-scale`;
- source frames: 83;
- selection: every fourth frame, maximum 21;
- labeled instances: 4;
- evaluated observations: 77;
- identity 3 is absent in 26 source masks.

Ground-truth masks generate boxes and IDs. Detector and segmentation errors are intentionally excluded.

## Results

### Controlled crossing

- oracle appearance: F1 1.000, 0 switches, 0 merges;
- RGB appearance: F1 1.000, 0 switches, 0 merges;
- V-JEPA appearance: F1 0.462, 2 switches, 2 merges;
- IoU only: F1 0.439, 3 switches, 1 merge;
- geometry only: F1 0.446, 2 switches, 2 merges;
- V-JEPA fused: F1 1.000, 0 switches, 0 merges.

### DAVIS `dogs-scale`

- oracle appearance: F1 1.000, 0 switches, 0 merges;
- RGB appearance: F1 0.374, 23 switches, 4 merges;
- V-JEPA appearance: F1 0.609, 16 switches, 4 merges;
- V-JEPA mask appearance: F1 0.513, 14 switches, 4 merges;
- IoU only: F1 0.768, 6 switches, 1 merge;
- default V-JEPA+IoU: F1 0.639, 4 switches, 3 merges.
- default mask-V-JEPA+IoU: F1 0.639, 4 switches, 3 merges.

V-JEPA improves appearance-only F1 by 0.235 over RGB and reduces switches by seven. Nevertheless, IoU remains stronger.
Default fusion trades fewer switches for more false merges and lower F1.

Hard mask-weighted pooling is a negative result: downsampling instance masks to the 24×24 V-JEPA grid reduces
appearance F1 by 0.096 and does not change fused performance. Higher-resolution/multi-layer projection is more promising
than hard final-grid masking.

## Sweep

Thirty same-sequence operating points vary appearance weight and threshold. The best F1 is 0.768, equal to IoU-only.
Because selection and evaluation use `dogs-scale`, this is exploratory and not held-out validation. The complete sweep is
stored in `metrics.json` and the W&B table.

## W&B

- run: `identity-ablation-real-vjepa`;
- promoted ID: `fw4rj25e`;
- URL: <https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/fw4rj25e>;
- content: dataset/variant metric table, F1 and switch comparisons, 30-point sweep table/scatter, scalar history, model
  configuration, timings, interactive HTML, metrics JSON, and Markdown record.

This run supersedes `neoffvmt` and `cfw43c0e`: it retains unambiguous `dataset/variant` labels and adds the hard-mask
pooling negative-result variants. Shared numerical results are unchanged.

## Artifacts

Ignored local output directory: `outputs/identity_ablation_davis/`.

- `metrics.json`: all defaults, sweeps, model config, timings, and provenance;
- `report.html`: interactive comparison and complete JSON payload;
- `EXPERIMENT.md`: generated concise record.

## Defect fixed

The original greedy association allowed multiple same-frame detections to enter one identity. The new frame-wise matcher
enforces one observation per track per view/time group. Unit tests reproduce the former merge opportunity and require two
separate tracks.

## Interpretation

The experiment provides evidence against using raw final-layer V-JEPA box pooling as a re-identification embedding. It
does show that V-JEPA carries more instance signal than simple RGB statistics. The controlled fusion success does not
transfer into a gain over IoU on `dogs-scale`.

The likely next improvement is mask-weighted, multi-layer V-JEPA pooling plus learned instance projection and global
motion-aware assignment—not another hand-tuned weighted sum.

## Limitations

- one real sequence;
- subsampled frames;
- same-sequence tuning;
- masks and boxes are ground truth;
- no real metric geometry;
- custom pairwise metrics rather than official DAVIS/HOTA tooling;
- CPU timing;
- no uncertainty calibration or confidence intervals.

## Next action

Freeze the operating point before evaluating additional DAVIS sequences. Implement mask-weighted multi-layer features,
constant-velocity prediction, and Hungarian assignment as separately ablated changes. Propagate merge/split evidence into
Phase 4 identity events without deleting historical observations.
