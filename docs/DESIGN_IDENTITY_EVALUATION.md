# Object identity evaluation and association ablations

## Motivation

Persistent memory is useful only if observations are assigned to the correct physical entity. Phase 4 persistence can
faithfully reload and replay a wrong association, so storage parity is not identity quality. This evaluation layer tests
the association boundary before planning or durable identity repair is expanded.

The central question is deliberately narrow: do V-JEPA2.1 box-pooled tokens improve same-category identity association,
and under what combination with motion overlap and geometry?

## Defect discovered in the initial clusterer

The initial Phase 3 greedy clusterer processed observations independently. Two detections with the same category,
view, and timestamp could both select the same cluster. That creates an impossible identity merge inside one frame.

The replacement groups observations by category, time, and view, scores every eligible track-observation pair, sorts
pairs by score, and accepts each track and observation at most once per group. Unmatched observations create new tracks.
Multi-view observations at the same time may still join because exclusivity is scoped to one view/time pair.

## Association score

The score is configurable through `AssociationConfig`:

```text
appearance = (cosine(mean(last four track embeddings), observation) + 1) / 2
iou        = box IoU with the latest track observation
geometry   = exp(-3D distance / geometry_distance_scale_m)

score = normalized weighted sum(appearance, iou, geometry)
```

Candidates below the threshold are rejected. Tracks older than `max_time_gap` cannot match. Weights are non-negative and
normalized; thresholds must be within `[0,1]`. Geometry contributes zero when either pose is absent rather than silently
inventing proximity.

This is still greedy bipartite matching, not globally optimal Hungarian/min-cost-flow tracking. The interface makes that
future replacement measurable.

## Controlled crossing fixture

The synthetic fixture renders two visually similar, same-category rectangular objects over ten frames. They move in
opposite directions, cross, disappear for different intervals, and re-enter. Every observation has a ground-truth ID,
mask, box, time, controlled appearance, and synthetic 3D position.

Its purpose is adversarial contract testing:

- one-to-one assignment in every frame;
- association across a multi-frame visibility gap;
- false merge/split and switch detection;
- comparison of isolated evidence sources;
- deterministic CI without downloads.

Synthetic colors are intentionally separable by raw RGB, so RGB and oracle appearance are upper-bound controls rather
than realistic baselines.

## Real-video fixture

The real experiment uses the official DAVIS 2017 480p train/validation archive and sequence `dogs-scale`. It contains four
same-category dog instances; identity 3 is absent in 26 of the 83 source annotations. The experiment selects every fourth
frame, capped at 21 frames, producing 77 labeled instance observations.

Images come from `JPEGImages/480p/dogs-scale`; ground-truth instance masks come from
`Annotations/480p/dogs-scale`. Boxes are computed from masks, so this experiment isolates association from detector and
mask errors. It does not measure the full Phase 3 detector pipeline.

Official source: <https://davischallenge.org/davis2017/code.html>.

Archive:

```text
DAVIS-2017-trainval-480p.zip
SHA-256 e3d0b5b77c3d031b000a19e0e25e3e2cac65d183755601bc2cf066df1a2aa492
```

Dataset files remain outside git.

## Feature variants

- `oracle_appearance`: one-hot ground-truth identity, an upper bound.
- `rgb_appearance`: crop channel statistics plus the existing deterministic category component.
- `vjepa_appearance`: real frozen V-JEPA2.1 ViT-B box-pooled patch tokens.
- `vjepa_mask_appearance`: the same tokens pooled only where the downsampled instance mask is active.
- `iou_only`: ambiguous appearance and box overlap only.
- `geometry_only`: controlled synthetic geometry only; DAVIS has no runtime geometry in this isolation.
- `vjepa_fused_default`: original synthetic 0.65/0.20/0.15 weighting.
- `vjepa_iou_default`: DAVIS 0.75/0.25 appearance/IoU with no fabricated geometry.
- `no_appearance`: controlled IoU/geometry baseline.

## Metrics

Predicted clusters induce same/different decisions over every observation pair:

- pairwise precision: predicted-same pairs that share ground-truth identity;
- pairwise recall: true-same pairs assigned to one predicted identity;
- pairwise F1: harmonic mean of pairwise precision and recall;
- ID switches: adjacent visible observations of a ground-truth identity assigned different tracks;
- fragments: additional predicted tracks used by one ground-truth identity;
- false merges: predicted tracks containing multiple ground-truth identities;
- track survival: fraction of each identity's observations in its largest predicted track;
- predicted track count and observation count.

These metrics are transparent approximations for the current fixture. Official HOTA/IDF1 evaluation remains future work.

## Controlled results

On 17 observations:

| Variant | Pairwise F1 | Switches | False merges | Tracks |
|---|---:|---:|---:|---:|
| Oracle appearance | 1.000 | 0 | 0 | 2 |
| RGB appearance | 1.000 | 0 | 0 | 2 |
| V-JEPA appearance | 0.462 | 2 | 2 | 2 |
| IoU only | 0.439 | 3 | 1 | 4 |
| Geometry only | 0.446 | 2 | 2 | 2 |
| V-JEPA fused default | 1.000 | 0 | 0 | 2 |
| No appearance | 0.439 | 3 | 1 | 4 |

The controlled fixture demonstrates complementarity, not generalization: V-JEPA alone fails, while fusion resolves the
designed ambiguity.

## DAVIS `dogs-scale` results

On 77 labeled observations:

| Variant | Pairwise F1 | Switches | False merges | Fragments | Survival |
|---|---:|---:|---:|---:|---:|
| Oracle appearance | 1.000 | 0 | 0 | 0 | 1.000 |
| RGB appearance | 0.374 | 23 | 4 | 8 | 0.536 |
| V-JEPA appearance | 0.609 | 16 | 4 | 8 | 0.708 |
| V-JEPA mask appearance | 0.513 | 14 | 4 | 8 | 0.702 |
| IoU only | 0.768 | 6 | 1 | 4 | 0.798 |
| V-JEPA+IoU default | 0.639 | 4 | 3 | 4 | 0.732 |
| V-JEPA mask+IoU default | 0.639 | 4 | 3 | 4 | 0.732 |

V-JEPA materially improves over RGB appearance, but appearance alone remains worse than IoU. Default fusion reduces
switches relative to IoU while increasing false merges and lowering pairwise F1.

Naive mask pooling does not fix the problem. At the 24×24 token grid, nearest-neighbor instance masks discard context and
can retain too few boundary tokens; appearance F1 falls from 0.609 to 0.513, while fused results are unchanged.

## Exploratory operating-point sweep

The experiment sweeps appearance weights `{0, .25, .5, .75, 1}` and thresholds `{.4, .5, .6, .7, .8, .9}` on the same
DAVIS sequence. The best pairwise F1 is 0.768 at several equivalent operating points, matching rather than exceeding the
IoU-only baseline. This is same-sequence selection and evaluation; it must not be called validation or held-out accuracy.

The result rejects a convenient hypothesis: simple weighted fusion of current box-pooled V-JEPA tokens does not improve
the best identity F1 on this sequence.

## Interpretation

V-JEPA tokens carry more identity information than the RGB-statistic baseline on similar real objects. However, their
box-pooled representation is not sufficiently instance-specific, and greedy weighted association turns some similarity
errors into persistent false merges. Motion overlap remains the strongest signal in this sequence.

Potential reasons:

- boxes include background and other overlapping dogs;
- V-JEPA is trained for semantic/temporal representation, not instance re-identification;
- tubelet features combine adjacent frames;
- absolute/spatial token content may dominate object texture;
- track embeddings average contaminated observations;
- no mask-weighted pooling or learned projection is used;
- greedy matching lacks motion prediction and global trajectory consistency.

## Next experiments

1. Compare higher-resolution and multiple V-JEPA layers instead of only final 24×24 tokens.
2. Train a compact instance projection with supervised contrastive/temporal losses.
3. Add Kalman/constant-velocity prediction and Hungarian assignment.
4. Test soft mask weighting and mask-plus-context rings rather than hard nearest-neighbor masks.
5. Use geometry/reprojection only when real camera/point uncertainty is available.
6. Evaluate multiple DAVIS sequences with a frozen operating point.
7. Add TAP-Vid/DAVIS official metrics and confidence intervals.
8. Feed identity events into Phase 4 split/merge/alias operations rather than overwriting history.

## Claim boundary

The supported claim is: real frozen V-JEPA2.1 box-pooled features contain more instance association signal than the simple
RGB baseline on one labeled DAVIS sequence, but the current default and swept weighted fusion do not outperform IoU-only
pairwise F1, and naive hard mask pooling is worse than box pooling. No claim is made about full DAVIS performance,
detector robustness, SAM2 tracking, or robot memory success.
