"""Single frozen protocol surface for formal Phase 2g-A.

All formal runners and selectors import this module so scientific constants do
not drift between Slurm array stages.
"""

from __future__ import annotations

from typing import Any

ARMS = ("M0", "M1", "M2", "M3")
CANDIDATES = ("M1", "M2", "M3")
FAMILIES = ("kv1", "kv2", "realsense", "xtion")
ROTATIONS: dict[str, dict[str, Any]] = {
    "R0": {"train": ("kv1", "xtion"), "validation": "realsense", "heldout": "kv2"},
    "R1": {"train": ("xtion", "realsense"), "validation": "kv2", "heldout": "kv1"},
    "R2": {"train": ("realsense", "kv2"), "validation": "kv1", "heldout": "xtion"},
    "R3": {"train": ("kv2", "kv1"), "validation": "xtion", "heldout": "realsense"},
}

SAMPLES_PER_FAMILY = 1024
EXPECTED_INVENTORY_COUNTS = {"kv1": 2003, "kv2": 3784, "realsense": 1159, "xtion": 3389}
QUALITATIVE_IDS_PER_FAMILY = 16
MINIMUM_VALID_PIXELS = 100
VALID_DEPTH_INTERVAL_M = (0.1, 10.0)
VIEWS_PER_TRAINING_SOURCE = 2
CAMERA_PROFILES = tuple(f"P{index}" for index in range(8))
CAMERA_QUALITY_PROFILES = tuple(f"P{index}" for index in range(1, 8))

LEARNING_RATES = (5e-4, 1e-3, 2e-3)
TUNING_SEED = 260629
TUNING_EPOCHS = 20
FORMAL_SEEDS = (0, 1, 2)
FORMAL_EPOCHS = 60
BATCH_SOURCE_GROUPS = 8
STEPS_PER_EPOCH = 256
TUNING_STEPS = 5_120
FORMAL_STEPS = 15_360
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 5.0
BOOTSTRAP_RESAMPLES = 100_000
BOOTSTRAP_SEED = 260629
IMAGE_SIZE = (384, 384)

EXPECTED_PARAMETERS = {"M0": 86_402, "M1": 92_820, "M2": 92_916, "M3": 93_685}

PRIMARY_METRICS = (
    "raw_abs_rel",
    "absolute_log_scale_error",
    "aligned_abs_rel",
    "nll",
    "ause",
)
ADDITIONAL_METRICS = (
    "signed_log_scale_error",
    "raw_rmse",
    "aligned_rmse",
    "delta1",
    "reliability_error",
    "coverage_50",
    "coverage_80",
    "coverage_90",
    "coverage_95",
)
LOWER_IS_BETTER = frozenset((*PRIMARY_METRICS, "raw_rmse", "aligned_rmse", "reliability_error"))

QUALITY_THRESHOLDS = {
    "raw_abs_rel_ratio_to_m0_max": 0.98,
    "absolute_log_scale_error_ratio_to_m0_max": 0.95,
    "aligned_abs_rel_ratio_to_m0_max": 1.02,
    "nll_difference_to_m0_max": 0.02,
    "ause_ratio_to_m0_max": 1.02,
    "raw_abs_rel_improving_families_min": 3,
    "raw_abs_rel_worst_family_ratio_to_m0_max": 1.05,
}
MECHANISM_THRESHOLDS = {
    "m2_raw_abs_rel_ratio_to_m1_max": 0.99,
    "m3_raw_abs_rel_ratio_to_m2_max": 0.99,
    "updated_to_control_raw_abs_rel_ratio_max": 0.99,
    "updated_control_improving_families_min": 3,
    "distinct_analytic_intrinsics_per_source": 8,
    "permutation_assignment_change_fraction": 1.0,
    "permutation_matrix_change_fraction": 1.0,
    "minimum_mean_absolute_prediction_delta_m_exclusive": 1e-6,
    "m3_full_to_zero_field_raw_abs_rel_ratio_max": 0.99,
    "m3_full_improving_families_min": 3,
}
RAW_ABS_REL_TIE_RELATIVE = 0.005
SCALE_ERROR_TIE_RELATIVE = 0.01
SIMPLICITY_ORDER = CANDIDATES
SELECTION_TIES = {
    "raw_abs_rel_relative": RAW_ABS_REL_TIE_RELATIVE,
    "absolute_log_scale_error_relative": SCALE_ERROR_TIE_RELATIVE,
    "simplicity_order": SIMPLICITY_ORDER,
}

MEMBERSHIP_SCHEMA = "jepa4d-phase2g-sun-membership-v1"
MATERIALIZATION_SCHEMA = "jepa4d-phase2g-sun-materialization-v1"
CACHE_RECEIPT_SCHEMA = "jepa4d-phase2g-cache-receipt-v1"
CACHE_AUDIT_SCHEMA = "jepa4d-phase2g-cache-audit-v1"
INPUT_SHARD_SCHEMA = "jepa4d-phase2g-sun-input-shard-v1"
FEATURE_SHARD_SCHEMA = "jepa4d-phase2g-sun-feature-shard-v1"
TARGET_SHARD_SCHEMA = "jepa4d-phase2g-sun-target-shard-v1"
ROTATION_VIEW_SCHEMA = "jepa4d-phase2g-rotation-view-v1"
METRICS_SCHEMA = "jepa4d-phase2g-metrics-v1"
NORMALIZATION_SCHEMA = "jepa4d-phase2g-rotation-feature-normalization-v1"
TUNING_RECEIPT_SCHEMA = "jepa4d-phase2g-tuning-run-v1"
LR_SELECTION_SCHEMA = "jepa4d-phase2g-lr-selection-v1"
FORMAL_TRAINING_RECEIPT_SCHEMA = "jepa4d-phase2g-formal-training-run-v1"
EVALUATION_RECEIPT_SCHEMA = "jepa4d-phase2g-heldout-evaluation-v1"
SELECTOR_SCHEMA = "jepa4d-phase2g-development-selector-v1"
WANDB_RECEIPT_SCHEMA = "jepa4d-phase2g-wandb-artifact-receipt-v1"

WANDB_ENTITY = "crlc112358"
WANDB_PROJECT = "jepa4d-worldmodel"
WANDB_GROUP_PREFIX = "phase2g-quality-"
CLAIM_BOUNDARY = "SUN RGB-D development evidence only; DIODE remains sealed and external final is unauthorized"


def expected_matrix_size(stage: str) -> int:
    if stage == "tuning":
        return len(ARMS) * len(ROTATIONS) * len(LEARNING_RATES)
    if stage in {"formal", "evaluation"}:
        return len(ARMS) * len(ROTATIONS) * len(FORMAL_SEEDS)
    raise ValueError(f"unknown Phase 2g matrix stage: {stage}")
