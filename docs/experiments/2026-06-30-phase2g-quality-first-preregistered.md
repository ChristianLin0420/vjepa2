# Phase 2g-A quality-first geometry — preregistered

## Status and authorization

**Status:** frozen and authorized for internal-research execution on SUN RGB-D. DIODE remains sealed and is not part of
this execution.

The project owner authorized internal SUN RGB-D training on 2026-06-30 under the citation conditions published by the
official SUN RGB-D project, with no raw-data redistribution. The repository records the detailed data-use decision in
the validation registry and its accompanying review record. Public availability is not represented as a standard SPDX
license; the authorization is a documented project-owner risk acceptance restricted to this internal research use.

This preregistration freezes Phase 2g-A before its first expanded-cache, tuning, formal-training, or held-out-evaluation
job. It incorporates Sections 1-15 of the Phase 2g proposal whose pre-execution SHA-256 is
`a245e8b0f9a041cd6841f313aad6097b3983d2f453f7de5736d8724fbbf66db2`. Where the proposal listed an item as proposed,
the exact values below are now frozen. Any semantic change to data membership, eligibility, model definitions, loss,
metrics, thresholds, array mapping, or selection creates a new protocol lineage and requires a new preregistration.

This protocol can support only a SUN RGB-D **development survivor**. It cannot support external confirmation,
deployment readiness, or a universal camera claim. Camera family is confounded with source and scene composition, and
the three seeds measure optimization variation rather than independent-camera uncertainty.

## Source and protected-data policy

The frozen source is SUN RGB-D V1 from `https://rgbd.cs.princeton.edu/`:

- archive filename: `SUNRGBD.zip`;
- bytes: `6,885,481,608`;
- SHA-256: `1a6dbf2a1c9044c4805a35ee648d616ea39a231fd5bd6f77e84cd2b8287fe41c`;
- families: `kv1`, `kv2`, `realsense`, and `xtion`;
- use: internal research only, citation required, no redistribution;
- raw RGB, depth, intrinsics, paths, protected archives, and large feature/target caches remain local and are never
  uploaded to W&B or committed to Git.

Cache job `C` never trusts or consumes a separately supplied extraction root. It re-hashes the opened official archive
file descriptor, selectively materializes exactly the direct leaf `image/*.jpg`, `depth_bfx/*.png`, and
`intrinsics.txt` files into fresh execution-scoped protected storage, and writes a relative per-file SHA-256 manifest.
The cache builder re-hashes the complete consumed tree against that archive-derived manifest before decoding any frame.

The frozen encoder is V-JEPA 2.1 ViT-B with the final 24x24 feature grid. Its checkpoint directory content identity is
`8c61f645d6252d619acdd15bca42f210fc27768050cc9995ebaa98cf6d779908`; the matched implementation identity is
`2479dbf282e31821dddfea7b8f26b4aee629b762c8fad4023d1f57a7e3f55d8c`. No teacher prediction or input checkpoint is
used by the Phase 2g heads.

DIODE is metadata-only and sealed. No Phase 2g worker may receive a DIODE/archive/devkit path or a value containing the
case-insensitive tokens `diode`, `external`, or `final` in a data-source field. The opacity jobs may inspect only the
previous Phase 2f seal receipts and the absence of an opening sentinel. They may not list, hash-stream, extract, decode,
load, cache, summarize, or visualize the DIODE archive.

## Frozen balanced membership and target separation

The cache stage enumerates the complete immutable SUN RGB-D leaf inventory and applies this deterministic procedure
before any model training:

1. Require exactly the published family inventories: 2,003 `kv1`, 3,784 `kv2`, 1,159 `realsense`, and 3,389 `xtion`
   leaves, each with one RGB image, one encoded depth image, and one intrinsics matrix.
2. Sort each family by the stable relative sample ID before opening depth.
3. In sorted order, decode only the candidate's depth and mark it mechanically eligible exactly when at least 100 pixels
   are finite and satisfy `0.1 < depth_m < 10.0`.
4. Select the first 1,024 eligible IDs from each family. Persist rejected sample ID plus boolean failure categories only;
   never persist a target value, histogram, distribution, preview, or aggregate target statistic in the membership
   artifact.
5. Abort if a family has fewer than 1,024 eligible IDs. Do not alter the threshold, backfill after observing model
   output, or silently omit a selected sample.
6. Select 16 qualitative IDs per family by the lowest `SHA256(sample_id)` before training.

This resolves the pre-access circularity in the proposal: the complete source identity and executable selection rule are
frozen here, while the exact selected-membership hash is produced atomically by cache job `C`. Architecture audit `Q`
must validate the 4x1,024 membership, selection trace, file identities, qualitative IDs, and cache hashes before any
tuning job can start. The membership is immutable for every downstream task and cannot depend on predictions or error.

Inputs, frozen features, and targets use separate per-family/per-profile shards. A training or tuning worker receives
only the two training-family target shards and one validation-family target shard for its rotation. It cannot resolve or
receive the held-out target shard. After checkpoint selection and hashing, a separate evaluation worker receives exactly
one held-out-family target shard. Cache and loader validators reject external-data fields and any unexpected target path.

The four rotations are frozen:

| Rotation | Training families | Validation family | Held-out development family |
|---|---|---|---|
| `R0` | `kv1`, `xtion` | `realsense` | `kv2` |
| `R1` | `xtion`, `realsense` | `kv2` | `kv1` |
| `R2` | `realsense`, `kv2` | `kv1` | `xtion` |
| `R3` | `kv2`, `kv1` | `xtion` | `realsense` |

Preprocessing is frozen to the proposal: 384x384 center-square V-JEPA input; a centered 0.85 second training crop;
96x96 RGB only where the scale path requires it; mask-weighted 24x24 target reduction with minimum valid mass 0.25;
finite target depths in `0.1 < depth < 10.0` metres; half-pixel `align_corners=False` crop/resize intrinsics; and feature
normalization fitted from each rotation's two training families only. Validation and ordinary held-out metrics use the
center-square view. P0-P7 camera profiles and the derangement `[5,6,3,2,1,7,0,4]` remain exactly as defined in the
proposal and Phase 2f preregistration.

## Models, losses, and health

All four arms train regardless of speed. Parameter counts are strict identity checks, never qualification limits:

| Arm | Trainable parameters | Frozen meaning |
|---|---:|---|
| `M0` | 86,402 | historical monolithic metric log-depth/log-variance decoder |
| `M1` | 92,820 | centered shape plus detached pooled-feature global scale |
| `M2` | 92,916 | M1 plus four canonical normalized intrinsics values |
| `M3` | 93,685 | M2 plus bounded, zero-mean, upsampled 4x4 scale field |

The Phase 2f loss weights, AdamW optimizer, weight decay `1e-4`, batch of eight source groups with two views, gradient
clip `5.0`, and bitwise-zero forbidden-gradient firewall are unchanged. There is no scheduler or early stopping. A run is
unhealthy only for a non-finite loss/gradient/output/metric, forbidden gradient above zero, absent required allowed
gradient, unchanged model state, strict reload/output mismatch, non-decreasing final-versus-initial 10% objective, or a
schema/source/dependency/W&B/artifact integrity failure. Quality, speed, memory, and parameter count cannot mark a run
unhealthy.

## Tuning, formal training, and immutable evaluation

Tuning runs all `4 arms x 4 rotations x 3 learning rates = 48` cells with seed `260629`, learning rates
`{5e-4, 1e-3, 2e-3}`, 20 epochs, and 5,120 optimizer steps per cell. Among healthy rates, select lower validation raw
AbsRel, then lower validation absolute log-scale error, then lower learning rate. Partial matrices cannot select.

Formal training runs all `4 arms x 4 rotations x 3 seeds = 48` cells for 60 epochs and 15,360 optimizer steps per cell,
using seeds `0,1,2` and the selected arm/rotation learning rate. Select the checkpoint with lowest validation raw AbsRel,
then lower validation absolute log-scale error, then earlier epoch. A failed cell remains a failure.

Exactly 48 separate immutable evaluation cells consume the selected checkpoints and corresponding held-out shards.
Training jobs never compute held-out metrics. Evaluation fits no point, scale, threshold, or variance parameter; it reuses
the positive variance multiplier fitted on validation pixels. Every same-checkpoint intervention is evaluation-only.

## Metrics, controls, and aggregation

The Phase 2g metric schema persists per-frame values and defines raw/aligned AbsRel, raw/aligned RMSE in metres, Delta-1,
signed and absolute log-scale error, validation-calibrated Gaussian log-depth NLL, AUSE/risk-coverage, empirical
50/80/90/95% coverage, reliability error, valid frames/pixels, typed failures, and predicted-versus-optimal scale
correlation/residuals. Formulas follow `docs/METRICS.md` and the frozen implementation tests; no old label may silently
change meaning.

The primary aggregate averages frames within each held-out family, averages the three seeds within its rotation, then
averages the four family means equally. A 100,000-resample paired hierarchical bootstrap uses seed `260629`, preserves
candidate/M0 pairing, averages optimizer seeds within a paired unit, resamples four family clusters with replacement,
then resamples frames within each selected family. Its interval is descriptive, not a population-significance claim.

M2/M3 evaluate P1-P7 under updated, stale, wrong, and permuted K. M0/M1 declare `consumes_intrinsics=false` and reject a
model call that passes K; their evaluator records `not_applicable_nonconsumer`. M3 additionally evaluates its selected
checkpoint with the scale field zeroed. No intervention retrains a model.

## Frozen development-survivor gates

A candidate must complete all 12 formal/evaluation cells and satisfy every condition versus paired M0:

- equal-family raw AbsRel ratio `<= 0.98`;
- absolute log-scale-error ratio `<= 0.95`;
- aligned AbsRel ratio `<= 1.02`;
- calibrated NLL difference `<= +0.02`;
- AUSE ratio `<= 1.02`;
- raw AbsRel improves in at least three of four held-out families;
- no held-out-family raw AbsRel exceeds `1.05x` paired M0.

M2 additionally requires raw AbsRel `M2/M1 <= 0.99`. M3 additionally requires `M3/M2 <= 0.99`. For M2/M3, updated K
must achieve raw AbsRel ratios `<= 0.99` versus each of stale, wrong, and permuted K, win in at least three families for
every control, have exactly eight distinct analytic K matrices per source, change permutation assignment and matrix 100%,
and produce a mean absolute prediction delta greater than `1e-6` metres per control. Full M3 must achieve raw AbsRel
`<= 0.99x` its zero-field intervention and improve in at least three families.

Among eligible candidates select lowest equal-family raw AbsRel. Within 0.5% relative, select lower scale error; within
1% relative scale error, select lower calibrated NLL; if still tied use `M1`, `M2`, `M3`. If none qualifies, retain M0.
Every selector output fixes `external_final_authorized=false`; Phase 2g-A cannot open DIODE.

## Slurm, W&B, and terminal evidence

The DAG is exactly 11 held base submissions and 152 logical tasks:

`T -> {O,C} -> Q -> H[0-47]%8 -> HG -> F[0-47]%8 -> V[0-47]%8 -> S -> G -> Z`, with `Q` depending on both `O` and
`C`. `T/O/C/Q/H/HG/F/V/S/G/Z` use the CPU, memory, time, and one-GPU allocations frozen in proposal Section 12. H, F,
and V are sequential arrays throttled to eight; the execution cannot exceed eight concurrent running allocations. Every
submission starts held until one atomic dependency graph records the clean commit, preregistration/source/sbatch hashes,
logical mappings, dependencies, resources, outputs, and failure semantics.

Real submission creates a dedicated execution worktree at the already-pushed commit and pins its tracking proof to an
immutable local anchor branch created only after the source branch equals its remote upstream. All job logs and formal
outputs live under the external execution state root, while the worktree/anchor are preserved as reproducibility
evidence through terminal `Z`; later edits or remote-branch movement cannot alter or invalidate the running checkout.
After a validated terminal `Z` SUCCESS receipt, the operator may remove the printed execution worktree with
`git worktree remove <execution_worktree>` and then delete the printed execution and pushed-anchor branches. Until that
receipt exists, all three are retained; an ambiguous interruption at scheduler release also retains them deliberately.

Online W&B is mandatory under entity `crlc112358`, project `jepa4d-worldmodel`, group
`phase2g-quality-<execution_id>`. Every logical task has one semantic run and receipt. Credentials, raw data, target
arrays, protected paths, CLI arguments, host metadata, and large caches are never serialized. W&B automatic metadata,
Git/code, console, machine-information, and system-stat capture are disabled; only explicit aggregate metrics,
sanitized artifacts, and numeric GPU telemetry are logged. SUCCESS is written last only after local validation and online artifact
completion. Terminal `Z` recursively validates all 152 task outcomes, scheduler accounting, graph edges, hashes, W&B
identities, completeness, DIODE opacity, and the selector's `external_final_authorized=false` before publishing terminal
status. Integrity pass and scientific promotion remain separate fields.

All retries retain failed scheduler history. Infrastructure retry is allowed only before trustworthy scientific output
exists. Any semantic code/data/protocol change requires a new lineage. Candidate underperformance is a completed negative
result, not grounds for changing a threshold or rerunning. A no-survivor result ends Phase 2g-A with M0 retained and DIODE
sealed.
