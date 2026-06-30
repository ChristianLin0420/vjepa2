# Phase 3 grounding and persistent-identity validation protocol

Status: **proposed, not executed**

Protocol date: 2026-06-30

Scope: labeled object grounding, instance segmentation, association-only identity, and end-to-end tracking

Research priority: quality and causal attribution first; latency and memory are recorded but are not quality-elimination gates

## 1. Decision this protocol must enable

The next Phase 3 study must answer four different questions without collapsing them into one score:

1. Can a text query select the correct object box?
2. Given the selected object, can the system predict an accurate instance mask?
3. Given perfect ground-truth observations, does the proposed appearance/geometry representation associate identities better?
4. When detections and masks are predicted, does the complete detector-segmenter-tracker improve persistent identity?

The following evidence ladder is mandatory:

```text
language -> predicted box -> predicted mask -> predicted observations -> persistent tracks
              |                |                     |
              |                |                     +-- end-to-end tracking evidence
              |                +-- segmentation evidence
              +-- grounding evidence

GT boxes/masks with hidden IDs -> association module -> association-only evidence
```

Association on ground-truth masks removes detector and segmenter errors by construction. It can validate an identity
representation or matcher, but it **cannot** validate an end-to-end tracker. Conversely, a weak end-to-end result does not
by itself show that the association representation is weak; detection recall, localization, mask quality, and association
must be reported separately.

## 2. Current evidence and its exact boundary

### 2.1 Grounding integration already completed

The completed Phase 3 run is integration evidence, not a labeled benchmark:

- the mock path used two generated 384x256 views containing colored rectangles and the queries `red mug` and
  `wooden table`;
- the full real-model path used the repository illustration `assets/architecture_vjepa2_1.jpg` with queries `person` and
  `robot`;
- real V-JEPA 2.1 ViT-B, VGGT-1B, and GroundingDINO-tiny executed end to end;
- the real illustration produced one `person` detection from two queries;
- masks were box rasters, not predicted instance masks;
- no labeled box, mask, negative-query, calibration, or temporal target existed;
- SAM2 was not installed or evaluated.

The deterministic smoke result of association recall 1.0, valid-mask fraction 1.0, and two expected slots is a software
contract only. The promoted W&B run `wvljbqlv` establishes model loading, typed evidence, persistence, visualization, and
logging—not grounding AP or segmentation IoU.

### 2.2 Identity ablation already completed

The current identity evidence has two parts:

- controlled fixture: two same-category objects, ten frames, crossing and disappearance/re-entry, 17 observations;
- DAVIS 2017 `dogs-scale`: 83 source frames, every fourth frame selected up to 21 frames, four labeled instances, and
  77 evaluated observations.

For `dogs-scale`, ground-truth masks supplied every box, mask, visibility decision, and object observation. Only IDs were
hidden from the matcher. The reported RGB, V-JEPA, IoU, geometry, switch, merge, and pairwise-F1 results therefore measure
**GT-mask association**, not detection, segmentation, or end-to-end tracking. The 30-point operating-point sweep used the
same sequence for selection and evaluation, so it is exploratory and consumed for that design claim. Its best point must
not be treated as a frozen operating point for new sequences.

## 3. Labeled datasets and assigned roles

No asset may enter a Slurm job until its source URL, byte count, SHA-256, annotation version, split file, access terms, and
redistribution policy are recorded in a manifest. A code license never automatically licenses the images or annotations.

### 3.1 Grounding and segmentation datasets

| Dataset | What it labels | Official split and proposed role | License/access treatment | Source |
|---|---|---|---|---|
| COCO 2017 detection/instance segmentation | 80 categories, boxes, and instance masks; more than 200,000 train/validation/test images overall | `train2017` is training-only if any detector or segmenter is fitted; `val2017` is development evaluation; hidden `test-dev` is optional one-shot confirmation after freezing. It is a category detection/segmentation benchmark, not a referring-expression benchmark. | Images retain their individual Flickr licenses; COCO annotations are governed by COCO terms. Record each downloaded archive and annotation identity, cite COCO, do not redistribute images in artifacts, and verify the current terms before execution. | [COCO 2017 task](https://cocodataset.org/dataset/detection-2017.htm), [COCO terms](https://cocodataset.org/#termsofuse), [official COCO API](https://github.com/cocodataset/cocoapi) |
| RefCOCO, UNC split | COCO object referents paired with natural-language expressions, COCO boxes, and COCO instance masks | 120,624/10,834/5,657/5,095 expression-region pairs for train/val/testA/testB; 16,994/1,500/750/750 images. Train fits learned projection; val selects thresholds/checkpoints; testA (people) and testB (non-people) are frozen formal splits. | Manual acquisition is required. The official REFER API is Apache-2.0, but that code license is not a blanket data license. Underlying COCO images retain their image-specific terms. Keep images local, do not upload them, and mark annotation redistribution `prohibited_pending_audit` unless the acquired package states otherwise. | [official REFER API/download instructions](https://github.com/lichengunc/refer), [dataset paper](https://arxiv.org/abs/1608.00272) |
| RefCOCO+, UNC split | Same task family, but absolute-location words were prohibited, making appearance/context more important | 120,191/10,758/5,726/4,889 expression-region pairs for train/val/testA/testB; 16,992/1,500/750/750 images. This is the primary language-grounding architecture test; roles match RefCOCO. | Same manual-access and underlying-COCO restrictions as RefCOCO. Freeze the exact UNC split; do not silently substitute Google/UMD splits or a third-party repack. | [official REFER API](https://github.com/lichengunc/refer), [RefCOCO/RefCOCO+ paper](https://arxiv.org/abs/1608.00272) |
| Flickr30K Entities | Caption phrases linked to manually annotated boxes on Flickr30K images | Independent-image Dataset B for phrase-localization transfer. Freeze the official train/validation/test split, but do not train, select, calibrate, or rewrite prompts on it; score the frozen COCO/RefCOCO-selected survivor once. | The annotation repository does not grant a blanket license for the underlying Flickr images. Audit current image/annotation terms, keep images local, and prohibit redistribution or W&B upload unless explicitly allowed. | [official annotation repository](https://github.com/BryanPlummer/flickr30k_entities), [benchmark paper](https://arxiv.org/abs/1505.04870) |

COCO and RefCOCO-family tasks share the COCO image source and together form Dataset A; they are complementary tasks, not
independent cross-dataset evidence. RefCOCO and RefCOCO+ testA/testB are separate estimands. A mean across them is reported
only after each split is shown; people/non-people failure cannot be hidden by pooling. Flickr30K Entities is Dataset B and
tests transfer to a different image/caption collection. If an official archive remains unavailable, the protocol stops
for that dataset instead of silently using a mirror with unknown content or terms.

### 3.2 Identity and tracking datasets

| Dataset | What it labels | Official split and proposed role | License/access treatment | Source |
|---|---|---|---|---|
| DAVIS 2017 semi-supervised video object segmentation | Dense per-object masks and persistent object IDs in multi-object videos | Official sets contain 60 train, 30 validation, 30 test-dev, and 30 test-challenge sequences. Train is representation/projection fitting; validation is local formal evaluation; test-dev is an optional one-shot server confirmation. The previously consumed `dogs-scale` sequence remains diagnostics-only and must be excluded from threshold/model selection regardless of its official partition. | Official direct downloads provide TrainVal images/annotations and first-frame annotations for test-dev/challenge. The cited download page does not state an unambiguous blanket data license, so record the acquired terms as `not_declared_on_download_page`, keep data internal, cite DAVIS, and prohibit redistribution pending legal review. | [official downloads](https://davischallenge.org/davis2017/code.html), [official rules/metrics](https://davischallenge.org/challenge2017/rulesdates.html), [2017 benchmark paper](https://arxiv.org/abs/1704.00675) |
| MOT17 | Pedestrian boxes and persistent track IDs; public detector sets and hidden test ground truth | Seven unique training sequences and seven unique test sequences. The official download repeats each sequence for DPM, FRCNN, and SDP detections; image frames must be deduplicated by base sequence. Training sequences fit/tune the matcher under a manifest-frozen sequence split; the hidden test server is one-shot confirmation. | Official data and development kit are downloadable from MOTChallenge. The data page identifies heterogeneous source videos rather than one blanket permissive license. Preserve source citations, accept current MOTChallenge terms, keep frames local, and prohibit redistribution in W&B/artifacts. | [official MOT17 data](https://motchallenge.net/data/MOT17/), [official MOT17 metrics/results](https://motchallenge.net/results/MOT17/), [MOTChallenge benchmark paper](https://arxiv.org/abs/2010.07548) |

DAVIS tests category-agnostic mask identity and re-entry. MOT17 tests crowded pedestrian detection and box identity. A
result on either dataset must not be generalized to arbitrary robot objects, multi-camera re-identification, or 3D identity.
Because both datasets participate in fitting/selection, they are complementary A1/A2 development suites, not an
independent identity-transfer pair. A later L2 claim requires a frozen matcher on a separately preregistered source such
as YouTube-VIS or PointOdyssey; until then, report only the named DAVIS/MOT17 development protocols.

## 4. Frozen split and target-access policy

1. Dataset manifests are generated before model-quality execution and include every selected sample/sequence ID.
2. RefCOCO/RefCOCO+ train and validation annotations are visible to training jobs; testA/testB annotations are visible
   only to the formal evaluator after checkpoint and threshold receipts are immutable.
3. COCO `val2017` is development evidence. If `test-dev` is used, only one frozen result submission is authorized.
4. Flickr30K Entities targets are unavailable to all fitting, selection, prompt-design, and calibration jobs. One frozen
   Dataset-A survivor is evaluated once; later tuning makes the result development-consumed and voids the transfer claim.
5. DAVIS train may fit a projection. DAVIS validation is formal local evidence. Test-dev is optional and requires a new
   authorization receipt; its server result cannot select another checkpoint.
6. MOT17 training sequences are divided deterministically at the sequence level, never the frame level. Hash-sort the
   seven base sequence IDs, assign the first five to fit/development and the final two to local validation, and persist
   the resulting IDs before decoding annotations. The official hidden test is evaluated once after freezing.
7. Frames from one video never appear in multiple fit/validation roles.
8. `dogs-scale` is tagged `consumed_exploratory` and cannot participate in a promotion aggregate.
9. No formal target may select a threshold, prompt template, non-maximum suppression rule, checkpoint, layer, or
   qualitative example.

## 5. Frozen systems and baselines

### 5.1 Grounding/segmentation matrix

| ID | Box source | Mask source | Appearance/evidence | Role |
|---|---|---|---|---|
| G0 | GroundingDINO-tiny | box raster | none | current deployable-component baseline |
| G1 | GroundingDINO-tiny | SAM2 prompted by predicted box | none | segmentation baseline; includes grounding error |
| G2 | Ground-truth referred box | SAM2 prompted by GT box | none | mask-model oracle-input diagnostic; never called end-to-end |
| G3 | GroundingDINO-tiny | SAM2 predicted-box mask | frozen final-layer V-JEPA box/mask evidence with current matcher | representation baseline |
| G4 | GroundingDINO-tiny | SAM2 predicted-box mask | proposed train-only standardized multi-layer V-JEPA projection | quality candidate |

All variants reuse identical text normalization, prompt templates, detector weights, detector thresholds, NMS, image
resolution, and SAM2 checkpoint within a comparison. G2 is an upper-bound decomposition row and cannot be promoted.

On COCO, GroundingDINO receives the exact category name list fixed in the manifest. On RefCOCO/+ and Flickr30K Entities,
it receives the exact human phrase without rewriting. Any prompt ensemble is a separate preregistered ablation.

### 5.2 Identity matrix

| ID | Observations | Identity cue/matcher | Evidence class |
|---|---|---|---|
| A0 | GT masks/boxes at every frame, IDs hidden | IoU-only Hungarian assignment | association-only baseline |
| A1 | GT masks/boxes at every frame, IDs hidden | RGB appearance only | association-only baseline |
| A2 | GT masks/boxes at every frame, IDs hidden | frozen final-layer V-JEPA appearance only | current representation baseline |
| A3 | GT masks/boxes at every frame, IDs hidden | IoU + final-layer V-JEPA + motion | strong association baseline |
| A4 | GT masks/boxes at every frame, IDs hidden | IoU + proposed multi-layer projection + motion | association-only candidate |
| E0 | predicted GroundingDINO boxes and SAM2 masks | IoU + motion | end-to-end baseline |
| E1 | same frozen predicted observations as E0 | IoU + final-layer V-JEPA + motion | end-to-end representation baseline |
| E2 | same frozen predicted observations as E0 | IoU + proposed multi-layer projection + motion | end-to-end candidate |

A0-A4 use the same perfect observations and differ only in association. E0-E2 use the same cached predictions so their
comparison isolates association under realistic observation errors. A separate detector/segmenter comparison may change
the observation cache, but it receives a new experiment ID and is not mixed with E0-E2.

For DAVIS semi-supervised VOS, also report S0 = frozen SAM2 first-frame propagation and S1 = candidate memory/association
on top of the same first-frame mask. This S0/S1 comparison follows the official DAVIS input condition. An E0-E2 DAVIS
run that discovers objects from predicted detections is a separate internal automatic-tracking protocol and must not be
submitted or labeled as official semi-supervised DAVIS. For MOT17, report both GT-box association A0-A4 and
private-detection end-to-end tracking E0-E2. Public-detector leaderboard rows are descriptive and must not be compared as
if their detector input were identical to the private GroundingDINO path.

## 6. Exact metrics

### 6.1 Grounding and mask metrics

For predicted region `P` and target `T`:

```text
IoU(P,T) = |P intersection T| / |P union T|
```

- **Box Acc@0.5**: fraction of referring expressions whose single top-scored predicted box has box IoU at least 0.5.
  Missing/abstained predictions score zero. This is primary for RefCOCO/+.
- **Box mean IoU**: arithmetic mean of top-1 box IoU over expressions, with missing predictions assigned zero.
- **Mask mIoU**: arithmetic mean of per-expression mask IoU; every expression has equal weight.
- **Mask oIoU**: sum of intersections across expressions divided by the sum of unions; large masks receive more weight.
- **Precision@0.5/0.7/0.9**: fraction of expressions whose mask IoU reaches each threshold.
- **Flickr30K Entities phrase localization**: Recall@1/5/10 at box IoU >= 0.5 using the frozen candidate ranking and the
  official phrase/split mapping; also report top-1 box Acc@0.5 so Dataset-A and Dataset-B behavior is readable together.
- **COCO box AP and mask AP**: official `COCOeval` AP averaged over IoU thresholds 0.50, 0.55, ..., 0.95 and 101 recall
  thresholds, category-aware, maximum 100 detections/image. Also report AP50, AP75, AP-small/medium/large and AR1/10/100.
  The official evaluator defines these settings in
  [`cocoeval.py`](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py).
- **Grounding calibration**: correctness is top-1 box IoU >= 0.5. Fit no parameters on formal data. Report Brier score
  `mean((confidence-correct)^2)` and 15-bin equal-width ECE
  `sum_b n_b/N * |accuracy_b-confidence_b|`. Empty bins contribute zero.
- **Coverage/selective risk**: at each fixed confidence threshold from 0.0 to 1.0 in steps of 0.05, report retained
  fraction and error `1-Acc@0.5` among retained queries. Abstention never disappears from the full-coverage score.

COCO AP, referring-expression Acc@0.5, mIoU, and oIoU answer different questions and must remain separate.

### 6.2 Association and tracking metrics

- **IDF1**: `2*IDTP / (2*IDTP + IDFP + IDFN)` after the benchmark's global identity matching.
- **MOTA**: `1 - (FN + FP + IDSW) / GT`. MOTA is detection-dominated and can be negative; it is secondary.
- **HOTA**: for localization threshold `alpha`, `HOTA_alpha = sqrt(DetA_alpha * AssA_alpha)`; report the official mean
  over alpha = 0.05, 0.10, ..., 0.95 plus DetA, AssA, DetRe, DetPr, AssRe, AssPr, and LocA. Use the official
  [TrackEval](https://github.com/JonathonLuiten/TrackEval) implementation and pin its commit. The metric definition is in
  the [HOTA paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7881978/).
- **ID switches**: official stricter MOTChallenge ID-switch count; also report switches per 1,000 GT detections.
- **Fragments**: number of interruptions of a ground-truth trajectory.
- **Mostly tracked/lost**: fractions covered for at least 80% / at most 20% of lifespan.
- **DAVIS J**: per-object region Jaccard/IoU.
- **DAVIS F**: per-object boundary F-measure under the official boundary matching tolerance.
- **DAVIS J&F**: arithmetic mean of object-averaged J and F, using the official DAVIS evaluator. Report mean, recall,
  and decay for J and F where supplied by the evaluator.
- **Current custom pairwise F1, false merges, and re-entry switches**: retain as diagnostics for continuity with the
  completed ablation, but never substitute them for HOTA/IDF1 or official DAVIS J&F.

In A0-A4, detection recall and mask accuracy are perfect inputs. HOTA/IDF1 from those rows are explicitly prefixed
`association_oracle_observations/`. In E0-E2, all detector false positives, false negatives, localization errors, mask
errors, and identity errors remain in the denominator and metrics are prefixed `end_to_end/`.

### 6.3 Aggregation and uncertainty

- RefCOCO/+ metrics: expression mean within split; show testA and testB separately, then their equal-split macro.
- Flickr30K Entities: phrase mean within the official split, category breakdowns where supplied, and image-clustered
  paired intervals so multiple phrases from one image are not treated as independent images.
- COCO: official category/image aggregation only; do not replace it with an image macro.
- DAVIS: official object aggregation, plus a table of every sequence and object.
- MOT17 local validation: official sequence concatenation for headline metrics, plus equal-sequence descriptive means.
- Three optimization seeds are run only for learned G4/A4/E2 components. Report mean and sample SD (`ddof=1`). Seeds are
  not independent datasets.
- Bootstrap intervals use 100,000 resamples with seed 260629 and resample the highest independent unit: images for
  static grounding and sequences for video tracking. Intervals are descriptive; hard gates use frozen point estimates
  and per-split/sequence protections.

## 7. Staged execution checklist

### Stage A — protocol, legal, and asset seal

- [ ] Freeze dataset versions, source URLs, citations, terms, redistribution fields, archive bytes, and SHA-256 hashes.
- [ ] Freeze all split/sample/sequence IDs and mark `dogs-scale` as consumed exploratory evidence.
- [ ] Pin V-JEPA, GroundingDINO, SAM2, COCO API, Flickr30K evaluator, DAVIS evaluator, TrackEval, CUDA, and environment identities.
- [ ] Freeze prompts, preprocessing, thresholds, NMS, Hungarian costs, maximum missed age, and random seeds.
- [ ] Write target-path denial tests for every formal split.

### Stage B — correctness and deterministic mini fixtures

- [ ] Run Ruff, mypy, unit tests, CUDA health, archive integrity, and online W&B preflight.
- [ ] Verify box/mask coordinate transforms numerically under resize, crop, and padding.
- [ ] Verify empty-query, no-detection, overlapping-mask, same-frame exclusivity, occlusion, exit/re-entry, and duplicate-ID cases.
- [ ] Prove COCO, Flickr30K phrase-localization, DAVIS, and TrackEval wrapper parity on tiny hand-computed fixtures.
- [ ] Require bitwise checkpoint reload and deterministic cached predictions under fixed hardware/software.

### Stage C — immutable observation caches

- [ ] Cache G0/G1 boxes, masks, scores, text, and image transforms for development splits.
- [ ] Cache DAVIS/MOT17 GT-observation rows with IDs encrypted/withheld from matcher code.
- [ ] Cache E0-E2 predicted observations once; every association arm consumes identical bytes.
- [ ] Record per-frame completeness, invalid masks, duplicate detections, and content hashes.

### Stage D — grounding and segmentation development

- [ ] Evaluate G0-G4 on RefCOCO/+ train/validation and COCO val2017.
- [ ] Fit G4 only on authorized train data and select its checkpoint/threshold only on validation.
- [ ] Run GT-box SAM2 diagnostic G2 without using it for promotion.
- [ ] Freeze one G4 checkpoint and operating point before formal test access.

### Stage E — association-only identity

- [ ] Evaluate A0-A4 on GT masks/boxes with IDs hidden.
- [ ] Fit A4 projection only on DAVIS train and MOT17 fit sequences.
- [ ] Select matching costs and age thresholds only on authorized development sequences.
- [ ] Freeze matcher/checkpoint before DAVIS validation and MOT17 local-validation aggregation.

### Stage F — end-to-end tracking

- [ ] Evaluate E0-E2 on identical predicted observations.
- [ ] Run DAVIS S0/S1 under official first-frame semi-supervised conditions.
- [ ] Preserve detector/mask errors in all E rows; never replace missed detections with GT observations.
- [ ] Freeze one complete candidate before any optional DAVIS test-dev or MOT17 test submission.

### Stage G — formal aggregation and release

- [ ] Open RefCOCO/+ formal annotations only after frozen receipts exist.
- [ ] If G4 is a development survivor, run the frozen G3/G4 pair once on Flickr30K Entities without retuning.
- [ ] Produce official and diagnostic metrics, intervals, per-split/sequence tables, and fixed qualitative panels.
- [ ] Apply the gates in Section 10 without post-result threshold changes.
- [ ] Upload only allowed artifacts; raw licensed images and annotations remain local.
- [ ] Run strict postflight over every receipt, dependency, hash, W&B run, and expected matrix cell.

## 8. Slurm execution and maximum concurrency

All model inference, training, and metric computation runs inside Slurm allocations. The login node may inspect hashes,
submit jobs, and read completed reports only.

```text
A asset/legal seal ----\
T tests/CUDA/W&B -------+--> C immutable caches --> D grounding array %8 ----\
                                                --> E oracle-ID array %8 -----+--> G aggregate
                                                --> F end-to-end array %8 ---/
```

- At most eight Slurm allocations may be simultaneously `RUNNING` across this protocol.
- Arrays use an explicit `%8` throttle. Independent arrays are linked so their combined runnable tasks cannot exceed eight.
- The submission controller records scheduler accounting and fails postflight if observed concurrency exceeds eight.
- Each semantic cell has one stable logical ID. Requeue retains that ID; it does not create a new scientific replicate.
- Every predeclared skipped cell writes a typed skip receipt and performs zero optimizer steps/inference.
- Dependencies use `afterok`; aggregators require the exact expected predecessor count and validate every parent SUCCESS,
  W&B receipt, file size, and SHA-256.
- No CPU fallback is allowed for a GPU-comparative cell. A hardware failure is infrastructure failure, not model evidence.

## 9. Fixed logging and visualization contract

Every scientific job uses online W&B with a unique run ID and uploads a small hash-bound artifact. Large feature caches,
licensed images, raw annotations, and credentials are never uploaded.

Required scalar/table namespaces:

- `grounding/<dataset>/<split>/box_acc50`, phrase R@K, box mIoU, mask mIoU/oIoU, P@threshold, COCO AP/AR, Brier, ECE, coverage;
- `association_oracle_observations/<dataset>/<split>/HOTA`, AssA, IDF1, IDSW, fragments, re-entry failures;
- `end_to_end/<dataset>/<split>/HOTA`, DetA, AssA, IDF1, MOTA, FP, FN, IDSW, fragments;
- `davis/<split>/J`, F, J&F, recall, decay;
- `runtime/<stage>/cold_ms`, `warm_ms`, p50/p95/p99, peak allocated/reserved memory, parameters, throughput;
- `integrity/*`: valid sample count, dropped count by typed reason, checkpoint/cache hashes, reload equality, expected jobs;
- full per-image/per-expression/per-sequence tables, not only aggregate summaries.

Required fixed visualizations:

1. RefCOCO/+ testA/testB grounding and mask bars plus Flickr30K transfer effects with seed SD.
2. COCO box/mask PR curves and AP by object size.
3. Confidence reliability and risk-coverage plots.
4. HOTA decomposition into DetA and AssA for oracle-observation and end-to-end rows in separate panels.
5. Per-sequence IDF1/HOTA forest plot with paired candidate-minus-baseline effects.
6. DAVIS J and F per-sequence plots plus temporal J/F decay.
7. ID-switch and fragmentation timeline for every selected qualitative video.
8. Error taxonomy: no detection, wrong referent, poor box, poor mask, false merge, false split, switch, lost/re-entry.
9. Resource table and stagewise latency timeline; these remain descriptive during quality selection.

Qualitative IDs are selected before inference by lowest SHA-256 within each dataset/split and failure-relevant stratum.
Use the same IDs for every variant. Never select “best”, “worst”, or surprising test examples after seeing targets. Each
panel shows input/query, GT box/mask/ID, predicted box/mask/ID, confidence, IoU, match decision, and typed failure.

## 10. Frozen promotion gates

All comparisons are paired on identical examples and observations where applicable. Percentage-point (`pp`) differences
are absolute differences on a 0–100 scale. Runtime and parameter count are reported but do not veto a scientifically
successful candidate in this phase.

### 10.1 Grounding/segmentation component gate: G4 over G3/G1

G4 is a **grounding/segmentation development survivor** only if all conditions hold:

1. RefCOCO+ equal-testA/B box Acc@0.5 is at least `+1.0 pp` over G3.
2. RefCOCO equal-testA/B box Acc@0.5 is no worse than G3 by more than `0.5 pp`.
3. RefCOCO+ mask mIoU is at least `+2.0 pp` over G1, and neither testA nor testB regresses by more than `1.0 pp`.
4. COCO val2017 box AP is no worse than the identical-detector G1 path by more than `0.5 pp`.
5. COCO val2017 mask AP is at least `+1.0 pp` over box-raster G0.
6. Brier score is lower than G3 and ECE is no worse by more than `0.01` on both RefCOCO and RefCOCO+.
7. Every expected example is scored; missing predictions remain errors; all values/checkpoints/hashes are finite and complete.

Passing this gate supports a labeled grounding/segmentation claim. It does not support persistent identity.

The separate **Dataset-B transfer confirmation** requires the frozen G4 to improve Flickr30K Entities top-1 Acc@0.5 and
R@1 over G3 by the effect margins fixed before targets are opened, with no prompt, threshold, checkpoint, or calibrator
change. Failure preserves only the Dataset-A development claim and prohibits a cross-dataset grounding claim.

### 10.2 Association-module gate: A4 over A3

A4 is an **association-only development survivor** only if all conditions hold on unconsumed sequences:

1. Equal-dataset mean IDF1 across DAVIS validation and MOT17 local validation is at least `+2.0 pp` over A3.
2. AssA is at least `+1.0 pp` on each dataset and HOTA is no worse on either dataset.
3. ID switches per 1,000 GT detections decrease by at least `10%` on the pooled validation evidence.
4. No sequence with at least 100 GT observations loses more than `5.0 pp` IDF1.
5. Same-frame exclusivity, one-to-one assignment, deterministic replay, and exact reload all pass.

Passing this gate supports only: “with ground-truth observations, the association module improves identity assignment.”
It does not promote E2 or establish detector-to-track quality.

### 10.3 End-to-end tracker gate: E2 over E1

E2 is the **Phase 3 tracker survivor** only if A4 passed and all conditions hold:

1. MOT17 local-validation IDF1 is at least `+1.0 pp` and HOTA at least `+0.5 pp` over E1.
2. MOT17 AssA improves by at least `+1.0 pp`; DetA differs from E1 by no more than `0.2 pp`, confirming identical
   observation input rather than detector drift.
3. MOT17 ID switches decrease by at least `5%`, with FP and FN exactly equal to E1 for the cached-observation comparison.
4. Under the internal automatic DAVIS protocol, E2 J&F improves by at least `+1.0 pp` over E1, with neither J nor F
   regressing. Separately, under official first-frame semi-supervised input, S1 J&F improves by at least `+1.0 pp` over
   S0, with neither J nor F regressing.
5. End-to-end IDF1 improves on at least 75% of evaluated validation sequences and no qualifying sequence regresses by
   more than `5.0 pp`.
6. Every expected frame, detection, mask, and track is present or represented by a typed miss; no GT repair is allowed.

Only E2 passing this gate permits the phrase “improved end-to-end tracking” on the named validation datasets. An optional
hidden-server result is confirmatory and cannot trigger another tuning cycle.

## 11. Claim boundaries

Allowed claims depend on the highest completed evidence tier:

| Highest completed tier | Allowed wording | Forbidden wording |
|---|---|---|
| Current repository illustration/mock | real-model integration is functional | grounding or segmentation accuracy |
| G4 component gate | improves grounding/segmentation on named RefCOCO/+/COCO splits | persistent identity, independent transfer, or open-world grounding |
| Flickr30K transfer gate | confirms frozen phrase grounding on an independent named image collection | unrestricted open-world or robot-domain grounding |
| A4 association gate | improves identity association given GT masks/boxes | improves end-to-end tracking |
| DAVIS S1 gate | improves semi-supervised video object segmentation on named DAVIS split | automatic detection or category grounding |
| E2 end-to-end gate | improves detector-to-track performance on named DAVIS/MOT17 validation protocols | arbitrary-object, robot-domain, 3D, multi-camera, or hardware reliability |
| Hidden server confirmation | confirms the frozen result on that hidden benchmark | population-wide or deployment-ready superiority |

Additional fixed limitations:

- RefCOCO-family images derive from COCO and may overlap foundation-model pretraining; overlap is unknown unless audited.
- Flickr30K Entities changes the image/caption collection but may also overlap foundation-model pretraining; it is a
  transfer benchmark, not proof of contamination-free generalization.
- RefCOCO/+ evaluates one referred target per expression and does not test unconstrained dialogue or negative queries.
- COCO category prompts do not measure free-form language grounding.
- DAVIS supplies first-frame masks in its semi-supervised protocol; this is not automatic object discovery.
- MOT17 is pedestrian-only, single-camera tracking and cannot establish generic robot-object identity.
- GT-mask association removes the principal observation errors and is an oracle-input diagnostic.
- Seed SD and bootstrap intervals do not turn a small number of datasets into population evidence.
- Geometry and 4D-memory benefits require a separately designed labeled 3D/multiview experiment; they are not inferred
  from 2D tracking gains.

## 12. Exit criteria and next handoff

Phase 3 validation is complete when the full predeclared matrix has either successful or typed skipped/failed receipts,
the strict postflight passes, and each survivor label is assigned according to Section 10. If only G4 passes, carry the
grounding/segmentation component forward without an identity claim. If A4 passes but E2 fails, keep the representation as
an association research result while retaining E1 operationally. Only an E2 pass authorizes integration into Phase 4
persistent memory as the new default observation-to-track path.
