#!/usr/bin/env python3
"""Materialize Phase 2g SUN inputs directly from the verified official archive."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from jepa4d.evaluation.phase2g_data import materialize_sun_archive
from jepa4d.training.phase2g_runtime import load_execution_provenance


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("formal Phase 2g SUN materialization may run only in Slurm")
    provenance = load_execution_provenance(args.provenance)
    dataset_root, receipt_path, receipt = materialize_sun_archive(
        args.archive,
        args.output,
        provenance=provenance,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "dataset_root": str(dataset_root),
                "receipt": str(receipt_path),
                "files_manifest_sha256": receipt["files_manifest_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
