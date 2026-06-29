from scripts.aggregate_phase2d_diagnostics import _camera_provenance, _control, _oracle_summary
from scripts.log_phase2d_diagnostics import _attribution_rows, _attribution_sequence_rows, _attribution_summary


def test_attribution_rows_flatten_visual_comparison_fields() -> None:
    record = {
        "seeds": [
            {
                "seed": 2,
                "interventions": [
                    {
                        "intervention": {
                            "intervention_id": "zero",
                            "final_coefficient": 1.0,
                            "effective_coefficients": {"2": 0.0, "5": 0.0, "8": 0.0},
                        },
                        "macro": {
                            "metric_abs_rel": 0.4,
                            "aligned_abs_rel": 0.2,
                            "metric_abs_log_scale_error": 0.3,
                            "calibrated_log_depth_nll": 1.1,
                            "prediction_delta_relative": 0.02,
                            "residual_total_norm_ratio": 0.0,
                        },
                    }
                ],
            }
        ]
    }
    rows = _attribution_rows(record)
    assert rows == [
        {
            "seed": 2,
            "intervention": "zero",
            "final_coefficient": 1.0,
            "layer_2": 0.0,
            "layer_5": 0.0,
            "layer_8": 0.0,
            "metric_abs_rel": 0.4,
            "aligned_abs_rel": 0.2,
            "abs_log_scale_error": 0.3,
            "calibrated_nll": 1.1,
            "prediction_delta_relative": 0.02,
            "total_residual_ratio": 0.0,
        }
    ]


def test_attribution_summary_flattens_control_list() -> None:
    record = {
        "aggregate": {
            "controls": [
                {
                    "intervention": {"intervention_id": "original"},
                    "metrics": {"metric_abs_rel": {"mean": 0.4, "std": 0.01, "values": [0.4]}},
                }
            ]
        }
    }
    assert _attribution_summary(record) == {"original/metric_abs_rel_mean": 0.4}


def test_attribution_sequence_rows_preserve_seed_sequence_intervention_identity() -> None:
    metrics = {
        "metric_abs_rel": 0.4,
        "aligned_abs_rel": 0.2,
        "metric_abs_log_scale_error": 0.3,
        "calibrated_log_depth_nll": 1.1,
        "prediction_delta_relative": 0.02,
        "residual_total_norm_ratio": 0.01,
    }
    rows = _attribution_sequence_rows(
        {
            "seeds": [
                {
                    "seed": 2,
                    "interventions": [
                        {
                            "intervention": {"intervention_id": "zero"},
                            "per_sequence": [
                                {"sequence_id": "freiburg3_long_office", "frames": 64, "metrics": metrics}
                            ],
                        }
                    ],
                }
            ]
        }
    )
    assert rows == [
        {
            "seed": 2,
            "intervention": "zero",
            "sequence_id": "freiburg3_long_office",
            "frames": 64,
            "metric_abs_rel": 0.4,
            "aligned_abs_rel": 0.2,
            "abs_log_scale_error": 0.3,
            "calibrated_nll": 1.1,
            "prediction_delta_relative": 0.02,
            "residual_total_norm_ratio": 0.01,
        }
    ]


def test_diagnostic_aggregate_helpers_keep_seed_and_sequence_boundaries() -> None:
    attribution = {
        "aggregate": {
            "controls": [{"intervention": {"intervention_id": "zero"}, "metrics": {"metric_abs_rel": {"mean": 0.4}}}]
        }
    }
    assert _control(attribution, "zero")["metrics"]["metric_abs_rel"]["mean"] == 0.4
    calibration = {
        "scale_oracle_audits": [
            {
                "audit_scope": "full_phase2c_test",
                "variant_id": f"seed{seed}:original",
                "oracles": {
                    "raw": {
                        "macro_equal_sequence_weight": {
                            "metric": {"abs_rel": 0.4 + seed * 0.01, "abs_log_scale_error": 0.3},
                            "aligned": {"abs_rel": 0.2},
                        }
                    }
                },
            }
            for seed in range(3)
        ]
    }
    summary = _oracle_summary(calibration)
    assert abs(summary["raw"]["metric_abs_rel"] - 0.41) < 1e-12


def test_camera_provenance_never_infers_registration_or_distortion_from_depth_status() -> None:
    calibration = {
        "calibration_audit": {
            "sequences": [
                {
                    "distortion": {"status": "unknown_not_declared"},
                    "rgb_depth_registration_status": "unknown_not_declared",
                    "depth": {
                        "provenance_status": "declared",
                        "duplicate_correction_status": "no_duplicate_detected",
                    },
                }
            ]
        }
    }
    provenance = _camera_provenance(calibration)
    assert provenance["overall"] == "incomplete"
    assert provenance["distortion"] == "incomplete"
    assert provenance["rgb_depth_registration"] == "incomplete"
    assert provenance["depth_correction"] == "declared_no_duplicate_detected"
