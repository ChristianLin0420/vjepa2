# Phase 1 representation validation specification

## Status and decision boundary

**Status:** proposed validation plan; not preregistered and not authorized for execution.

This document defines the model-quality gate that must follow the completed Phase 1 integration work. It does not change
the promoted Phase 1 record or claim that a labeled representation benchmark has already run.

The objective is to determine whether frozen V-JEPA 2.1 tokens encode useful spatial and temporal information across two
independent video domains, whether a different layer policy improves that information, and whether any observed gain is
caused by temporal evidence rather than data leakage or probe capacity. Architecture quality is selected first. Runtime,
throughput, memory, and artifact size are mandatory diagnostics but cannot eliminate a representation candidate before a
quality winner is frozen.

## 1. Current actual evidence

The current promoted evidence is **integration**, not representation accuracy:

- the real V-JEPA 2.1 ViT-B path emitted finite `[1,1,4,576,768]` tokens for an eight-frame generated video;
- layers 2, 5, 8, and 11, PyTorch/Zarr artifacts, PCA, token histograms, temporal cosine traces, and W&B artifacts were
  produced successfully;
- the promoted run is [W&B `gisjdqvx`](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/gisjdqvx), with the durable
  record in [the Phase 1 experiment](../experiments/2026-06-28-phase1-initial.md);
- the observed adjacent-bin cosine of about `0.9955` came from a smooth generated RGB sequence. It is a finite/noncollapse
  diagnostic, not labeled temporal-understanding evidence;
- Phase 2b later showed that frozen final-layer V-JEPA features support a compact TUM depth probe, but that is downstream
  geometry evidence and does not replace a Phase 1 action/temporal benchmark.

No Something-Something V2 or EPIC-KITCHENS-100 training, validation, official-test submission, or labeled comparison has
been executed in this project. No current number supports broad semantic, action-recognition, anticipation, retrieval, or
cross-domain representation claims.

## 2. Dataset roles, access, and licensing

Only official downloads and annotation repositories may populate a manifest. Mirrors may be used to verify availability,
but their metadata or claimed license must not override the dataset owner's terms.

### Dataset A1 — Something-Something V2

**Role:** primary temporal-reasoning benchmark. Its short object-interaction clips and fine-grained action labels make
frame order, state change, and direction important, so it is the main test of tubelet-aware frozen features.

Primary sources:

- [official Qualcomm download page](https://www.qualcomm.com/developer/software/something-something-v-2-dataset/downloads);
- [official download instructions](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/20bn-something-something_download_instructions_-_091622-v2.pdf);
- [dataset paper](https://arxiv.org/abs/1706.04261).

Access and license caveats:

- the official host requires all video parts, instructions, and label package to be downloaded and assembled together;
- the public download page does not state a simple open-data license inline. Before staging, archive the exact license or
  click-through terms presented by Qualcomm, record the retrieval date and hashes, and obtain project approval;
- absence of an inline license is not permission to redistribute. Raw clips, decoded frames, participant information, or
  label packages must not be uploaded to W&B or committed to Git;
- if the accepted terms are incompatible with the intended use, stop this dataset branch and record it as unavailable;
  do not substitute an unverified mirror silently.

### Dataset A2 — EPIC-KITCHENS-100 complementary development

**Role:** complementary egocentric action-anticipation development. It tests a different labeled capability on long,
unscripted, wearer-centric kitchen activity and unseen participants, but because it participates in architecture selection
it is not an independent Dataset-B transfer confirmation.

Primary sources:

- [official EPIC-KITCHENS dataset site](https://epic-kitchens.github.io/2025);
- [official EPIC-KITCHENS-100 annotations and split files](https://github.com/epic-kitchens/epic-kitchens-100-annotations).

Access and license caveats:

- the official site publishes the dataset under **CC BY-NC 4.0**; attribution, license linkage, change disclosure, and
  noncommercial-use restrictions apply. Commercial use requires a separate license from the owners;
- the videos contain activity in participants' homes. Treat raw video and participant IDs as restricted research data;
- the official site reports an erratum affecting pre-extracted RGB/flow frames for two videos. Prefer source videos and
  deterministic local extraction, or pin and document the official correction before manifest freeze;
- pin the annotations repository commit and hash every train/validation/test timestamp file used. W&B receives metrics
  and approved derived diagnostics, never raw home video.

## 3. Frozen split and leakage policy

The exact subset size, preprocessing, probe code, seeds, and numerical gates must be converted into a preregistration
before jobs are submitted. The following policy is fixed conceptually:

1. Preserve each dataset's official train, validation, and test identities. Never move a clip, video, participant, or
   kitchen across official boundaries.
2. Fit probe parameters and feature normalization using official training data only; derive label maps mechanically from
   the official training schema rather than observed validation/test labels.
3. Use official validation data for learning-rate/checkpoint selection and any temperature, confidence, or threshold
   calibration. Freeze the calibrator before held-out scoring.
4. Keep official test labels and challenge/server feedback unavailable to training and selection workers. Make at most one
   test submission per frozen winner under a separate authorization. If a current official test service is unavailable,
   report validation evidence and do not relabel it as test evidence.
5. Preserve EPIC participant/kitchen grouping. Report seen/unseen participant and tail-class slices using only official
   annotations; do not create random clip-level splits that leak one source video across partitions.
6. Freeze manifests before feature extraction. Each entry records dataset version, source URL, license snapshot identity,
   sample/video ID, official split, media bytes/SHA-256, annotation bytes/SHA-256, timestamps, and decode status.
7. Deduplicate by source video and content hash across train/validation/test. A collision is a hard failure.
8. Select qualitative examples by sample-ID hash before labels, predictions, or errors are inspected.

## 4. Models, probes, and baselines

All learned comparisons use the same decoded clips, augmentation budget, frozen encoder policy, probe family, optimizer
search, number of epochs, seeds, and validation selector.

| ID | Role | Frozen representation | Trainable component |
|---|---|---|---|
| `R0` | trivial floor | none; class prior | none |
| `R1` | untrained-control floor | randomly initialized frozen ViT-B with the same token shape | registered probe only |
| `R2` | non-JEPA visual baseline | preregistered framewise image encoder or RGB video baseline | capacity-matched probe |
| `R3` | operational reference | V-JEPA 2.1 ViT-B normalized final layer | registered probe |
| `R4` | fixed hierarchy ablation | fixed mean of layers 2/5/8/11 | identical probe |
| `R5` | candidate | learned train-only layer fusion over 2/5/8/11 | fusion plus the same probe |

An official V-JEPA 2 checkpoint may be reported as a historical external reference only when its size, resolution,
preprocessing, and clip policy are named. A capacity-mismatched checkpoint cannot govern promotion.

The primary protocol is a frozen encoder plus a compact temporal probe. Any selective-unfreezing experiment is a separate
ablation and cannot be pooled with frozen-probe results. Probe parameter counts must match across `R1`-`R5`; the learned
fusion parameters are reported separately.

Same-checkpoint interventions are mandatory:

- correct order versus deterministic frame shuffle;
- correct order versus temporal reversal;
- full clip versus registered middle-frame repetition;
- correct view/time identity versus identity disabled.

Interventions are evaluated without retraining. A prediction change demonstrates sensitivity; a quality drop under a
destructive intervention supports useful temporal dependence.

## 5. Metrics and aggregation

Use the official evaluator where one exists and persist per-example results.

### Something-Something V2

- primary: top-1 action accuracy;
- secondary: top-5 accuracy, mean-class accuracy, per-class recall, confusion matrix, and negative log-likelihood;
- mechanism: paired top-1 change under shuffle, reversal, repeated-frame, and disabled-time-identity controls;
- aggregation: one prediction per official clip, then clip macro and class macro; report three-seed mean and sample SD.

### EPIC-KITCHENS-100

- primary: official action-anticipation class-mean top-5 recall;
- secondary: verb, noun, and action top-5 recall; overall, unseen-participant, and tail-class results; NLL and expected
  calibration error where logits are available;
- aggregation: use the official annotation/evaluation implementation and participant-aware slices; report three-seed
  mean and sample SD without treating seeds as independent kitchens.

### Cross-dataset and diagnostic quantities

- paired candidate-minus-`R3` effects with a preregistered hierarchical bootstrap over clips nested in source videos;
- feature finite fraction, token norm, per-layer variance/effective rank, and collapse alarms;
- trainable/total parameters, throughput, p50/p95 latency, peak allocated/reserved memory, and artifact size.

Accuracy and calibration select the architecture. Efficiency is descriptive until the quality winner is frozen.

## 6. Promotion gates

The preregistration must freeze the following margins before any labeled training. Proposed margins are shown here to
make the intended decision explicit.

### Execution gate

- every declared dataset/model/seed cell completes or is reported as a failure;
- all outputs and metrics are finite, checkpoints reload exactly on a fixed batch, and manifests/hashes validate;
- validation alone selects hyperparameters and checkpoints;
- all W&B runs finish online with unique local receipts and durable local artifacts;
- no raw licensed media appears in Git, W&B, reports, or model artifacts.

### Representation-quality gate

`R3` establishes Phase 1 model-quality evidence only if it beats `R1` and the strongest registered non-JEPA baseline on
both primary development endpoints and its paired performance falls under every destructive temporal control. A learned
candidate (`R4` or `R5`) is promoted over `R3` only if all conditions hold:

- Something-Something V2 top-1 improves by at least **1.0 absolute percentage point**;
- EPIC action-anticipation class-mean top-5 recall improves by at least **0.5 absolute percentage point**;
- no primary verb, noun, action, unseen-participant, tail-class, or mean-class metric regresses by more than **0.5 point**;
- validation-calibrated NLL does not worsen by more than the preregistered tolerance;
- the same candidate wins or ties directionally in at least two of three seeds on each dataset;
- temporal interventions produce the expected degradation and no leakage/integrity check fails.

If no candidate qualifies, retain the final-layer `R3` reference. Do not promote a faster but less accurate candidate, and
do not change endpoints after results are visible. Optimize the frozen winner in a subsequent parity-tested experiment.

## 7. W&B and durable visual logging

Every logical run has one unique online W&B run and artifact receipt. Required panels include:

- train/validation loss, primary/secondary metrics, learning rate, gradients, and checkpoint rank;
- dataset/model/seed comparison tables and paired effects;
- per-class recall and confusion matrices;
- unseen-participant and tail-class slices;
- correct/shuffled/reversed/repeated-frame intervention deltas;
- fixed-ID RGB contact sheets, token PCA, temporal similarity traces, and layer statistics;
- calibration/reliability plots and failure counts;
- descriptive latency, throughput, memory, utilization, temperature, power, and clock telemetry.

Durable local outputs include versioned JSON/JSONL, CSV, NPZ, checkpoints, manifests, PNG, self-contained HTML, logs,
receipts, and SHA-256 identities. Raw video is neither an artifact nor a visualization payload.

## 8. Slurm and resource policy

- all GPU extraction, training, and evaluation runs through Slurm; the login node is limited to inspection, metadata-only
  staging, environment construction, submission, and monitoring;
- use held dependency graphs and semantic job names; arrays replace hundreds of independent submissions;
- every GPU array is throttled with `%8`; extraction, tuning, formal training, and evaluation arrays run sequentially so
  no path can exceed **eight concurrent RUNNING allocations**;
- scalar audit/selector jobs depend on the arrays and must not overlap in a way that raises the cap;
- record requested/actual partition, node, GPU, wall time, exit state, requeues, and accounting-derived peak concurrency;
- retry only infrastructure failures that produced no trustworthy scientific result, preserving the same logical identity
  and complete scheduler history.

## 9. Staged TODO

### A. Legal and data freeze

- [ ] Capture and approve the current Qualcomm Something-Something V2 license/download terms.
- [ ] Capture EPIC-KITCHENS CC BY-NC 4.0 terms, attribution, annotation commit, and frame erratum.
- [ ] Build checksum-pinned official manifests and run split/content deduplication.
- [ ] Freeze decode, clip sampling, augmentation, qualitative IDs, and test-server policy.

### B. Adapter and protocol freeze

- [ ] Implement official-label adapters and unit-test sample/time identity.
- [ ] Freeze probe architecture, layer policies, controls, seeds, search space, epochs, and metric schema.
- [ ] Validate exact checkpoint reload and media-exclusion checks on a tiny train-only fixture.
- [ ] Convert this proposal into a hash-bound preregistration.

### C. Development execution

- [ ] Extract frozen features under the max-eight Slurm DAG.
- [ ] Run equal-budget tuning and three-seed formal training for `R1`-`R5`.
- [ ] Evaluate validation metrics, calibration, temporal controls, failures, and resource diagnostics.
- [ ] Complete content, scheduler, W&B, and artifact postflight before selection.

### D. Selection and confirmation

- [ ] Apply the frozen quality gates and select one candidate or retain `R3`.
- [ ] Freeze the selected checkpoint, calibration, code, and manifest identities.
- [ ] Request separate authorization for one official-test submission, if an official service is available.
- [ ] Optimize runtime only after winner freeze and require prediction/metric parity.

## 10. Claim boundary

A completed development run may establish frozen-probe representation quality on the named Something-Something V2 and
EPIC-KITCHENS-100 protocols. It does not establish universal video understanding, robot-policy quality, causal physical
reasoning, independent cross-dataset transfer, or commercial deployment rights. A validation-only result is not an official-test result. Three optimizer seeds
are not three independent datasets. Temporal sensitivity is not useful temporal understanding unless the correct-order
prediction also improves target-scored quality. Hardware timing remains specific to the frozen device, precision, batch,
software, and measurement protocol.
