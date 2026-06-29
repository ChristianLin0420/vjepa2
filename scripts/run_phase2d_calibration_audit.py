#!/usr/bin/env python3
"""Run the CPU-only Phase-2d calibration and scale-oracle audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jepa4d.evaluation.phase2d_calibration_audit import (
    assemble_phase2d_report,
    audit_tum_manifest,
    build_manifest_intrinsics_controls,
    discover_phase2c_prediction_files,
    load_prediction_sets,
    run_scale_oracle_audit,
    write_phase2d_outputs,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit TUM K/crop/FoV/depth provenance and run target-fitted diagnostic scale oracles on persisted depth arrays."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Phase-2c TUM cross-sequence manifest")
    parser.add_argument(
        "--dataset-parent",
        type=Path,
        help="Optional extracted TUM parent; enables selected-file image-size, transformed-K, and FoV checks",
    )
    parser.add_argument(
        "--phase2c-output",
        type=Path,
        help="Optional Phase-2c output. Its compact diagnostics/*.npz are used as a visibly labeled fallback.",
    )
    parser.add_argument(
        "--predictions-npz",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional full or compact prediction NPZ; repeat for multiple files. Full files should use "
            "jepa4d-phase2d-depth-predictions-v1."
        ),
    )
    parser.add_argument("--output", type=Path, required=True, help="New audit output directory")
    parser.add_argument("--spatial-grid-height", type=int, default=4)
    parser.add_argument("--spatial-grid-width", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    prediction_paths = [path.resolve(strict=True) for path in args.predictions_npz]
    if args.phase2c_output is not None:
        prediction_paths.extend(discover_phase2c_prediction_files(args.phase2c_output.resolve(strict=True)))
    prediction_paths = list(dict.fromkeys(prediction_paths))

    manifest = args.manifest.resolve(strict=True)
    dataset_parent = args.dataset_parent.resolve(strict=True) if args.dataset_parent is not None else None
    calibration = audit_tum_manifest(manifest, dataset_parent=dataset_parent)
    controls = build_manifest_intrinsics_controls(calibration)
    prediction_sets = load_prediction_sets(prediction_paths, manifest_path=manifest)
    oracle_audits = [
        run_scale_oracle_audit(
            prediction_set,
            spatial_grid_size=(args.spatial_grid_height, args.spatial_grid_width),
        )
        for prediction_set in prediction_sets
    ]
    report = assemble_phase2d_report(
        manifest_path=manifest,
        calibration_audit=calibration,
        oracle_audits=oracle_audits,
        intrinsics_controls=controls,
        prediction_paths=prediction_paths,
    )
    paths = write_phase2d_outputs(report, args.output)
    print(
        json.dumps(
            {
                "result": "success",
                "diagnostic_only": True,
                "prediction_sets": len(prediction_sets),
                "audit_scopes": report["audit_scopes"],
                "outputs": paths,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
