#!/usr/bin/env python3
"""Prepare and strictly audit the SUN RGB-D sensor-blocked manifest on a login node."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jepa4d.benchmarks.geometry.sun_rgbd import (
    DEFAULT_TARGET_COUNTS,
    OFFICIAL_ARCHIVE_BYTES,
    OFFICIAL_ARCHIVE_SHA256,
    audit_manifest_summary,
    build_sensor_blocked_manifest,
    load_sensor_blocked_manifest,
    write_sensor_blocked_manifest,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the official SUNRGBD.zip, enumerate strict sensor leaves, generate the deterministic sensor-blocked "
            "manifest, and independently reload/hash/decode every selected frame. CPU/login-node only."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="Extracted .../SUNRGBD directory")
    parser.add_argument("--archive", type=Path, required=True, help="Official SUNRGBD.zip")
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--kv1-count", type=int, default=DEFAULT_TARGET_COUNTS["kv1"])
    parser.add_argument("--xtion-count", type=int, default=DEFAULT_TARGET_COUNTS["xtion"])
    parser.add_argument("--realsense-count", type=int, default=DEFAULT_TARGET_COUNTS["realsense"])
    parser.add_argument("--kv2-count", type=int, default=DEFAULT_TARGET_COUNTS["kv2"])
    parser.add_argument(
        "--clamp-max-depth-m",
        type=float,
        default=8.0,
        help="Explicit protocol clamp. Use --no-depth-clamp to preserve all decoded values.",
    )
    parser.add_argument("--no-depth-clamp", action="store_true")
    parser.add_argument("--expected-archive-bytes", type=int, default=OFFICIAL_ARCHIVE_BYTES)
    parser.add_argument("--expected-archive-sha256", default=OFFICIAL_ARCHIVE_SHA256)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_root = args.dataset_root.resolve(strict=True)
    archive = args.archive.resolve(strict=True)
    clamp = None if args.no_depth_clamp else args.clamp_max_depth_m
    target_counts = {
        "kv1": args.kv1_count,
        "xtion": args.xtion_count,
        "realsense": args.realsense_count,
        "kv2": args.kv2_count,
    }
    manifest = build_sensor_blocked_manifest(
        dataset_root,
        archive,
        target_counts=target_counts,
        clamp_max_depth_m=clamp,
        expected_archive_bytes=args.expected_archive_bytes,
        expected_archive_sha256=args.expected_archive_sha256,
        verify_archive_hash=True,
    )
    write_sensor_blocked_manifest(manifest, args.manifest_output)
    bundle = load_sensor_blocked_manifest(
        dataset_root,
        args.manifest_output,
        verify_file_hashes=True,
        validate_depth=True,
    )
    audit = audit_manifest_summary(bundle)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
