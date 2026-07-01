from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from scripts.write_phase2g_dependency_graph import _parse_job, parse_sbatch
from slurm.phase2g_contract import (
    ACCOUNT,
    LEARNING_RATES,
    PARTITIONS,
    SUBMISSION_POLICY,
    array_coordinates,
    expected_labels,
    parent_map,
)

ROOT = Path(__file__).resolve().parents[2]
SUBMITTER = ROOT / "slurm" / "submit_phase2g.sh"


def test_exact_152_task_dag_and_only_t_is_root() -> None:
    labels = expected_labels()
    parents = parent_map()
    assert len(labels) == 152
    assert set(parents) == labels
    assert [label for label, values in parents.items() if not values] == ["T"]
    assert len([label for label in labels if label.startswith("H-")]) == 48
    assert len([label for label in labels if label.startswith("F-")]) == 48
    assert len([label for label in labels if label.startswith("V-")]) == 48
    assert parents["Q"] == ["O", "C"]
    assert parents["G"] == ["S"]
    assert parents["Z"] == ["G"]
    assert SUBMISSION_POLICY == {
        "all_jobs_submitted_held": True,
        "dependency_type": "afterok",
        "only_root_without_dependency": "T",
        "release_after_atomic_graph_write": True,
        "logical_job_count": 152,
        "scheduler_submission_count": 11,
        "max_parallel_tasks": 8,
        "array_task_throttle": 8,
        "external_final_authorized": False,
    }


def test_array_mapping_is_bijective_and_stage_specific() -> None:
    for stage, prefix in (("tuning", "H"), ("formal", "F"), ("evaluation", "V")):
        rows = [array_coordinates(stage, task) for task in range(48)]
        assert len({row["label"] for row in rows}) == 48
        assert all(str(row["label"]).startswith(f"{prefix}-") for row in rows)
    tuning = [array_coordinates("tuning", index) for index in range(48)]
    assert [row["learning_rate"] for row in tuning[:3]] == list(LEARNING_RATES)
    assert array_coordinates("formal", 0) == {"arm": "M0", "rotation": "R0", "seed": 0, "label": "F-M0-R0-S0"}
    assert array_coordinates("evaluation", 47) == {
        "arm": "M3",
        "rotation": "R3",
        "seed": 2,
        "label": "V-M3-R3-S2",
    }
    with pytest.raises(ValueError):
        array_coordinates("formal", 48)


def test_base_sbatch_resources_match_frozen_proposal() -> None:
    expected = {
        "phase2g_tests.sbatch": ("16", "160G", "01:30:00"),
        "phase2g_opacity.sbatch": ("8", "32G", "00:30:00"),
        "phase2g_cache.sbatch": ("16", "160G", "04:00:00"),
        "phase2g_audit.sbatch": ("8", "64G", "01:00:00"),
        "phase2g_array_dispatch.sbatch": ("16", "160G", "04:00:00"),
        "phase2g_lr_select.sbatch": ("8", "64G", "01:00:00"),
        "phase2g_select.sbatch": ("16", "64G", "02:00:00"),
        "phase2g_external_guard.sbatch": ("8", "32G", "00:30:00"),
        "phase2g_postflight.sbatch": ("16", "64G", "02:00:00"),
    }
    for name, values in expected.items():
        resources = parse_sbatch(ROOT / "slurm" / name)
        directives = resources["directives"]
        assert directives["account"] == ACCOUNT
        assert directives["partition"] == PARTITIONS
        assert (directives["cpus-per-task"], directives["mem"], directives["time"]) == values
        assert (directives["nodes"], directives["ntasks"], directives["gres"]) == ("1", "1", "gpu:1")


def test_submitter_has_exact_held_dependency_and_dry_run_contract() -> None:
    source = SUBMITTER.read_text(encoding="utf-8")
    assert os.access(SUBMITTER, os.X_OK)
    assert source.count("--hold") == 1
    assert '--dependency "afterok:$dependency"' in source
    assert "--array=0-47%8" in source
    assert source.count("--array=0-47%8") == 3
    assert source.count('F="$(submit "$name" "$HG"') == 1
    assert "[[ ${#submission_ids[@]} -eq 11 ]]" in source
    assert "[[ ${#graph_jobs[@]} -eq 152 ]]" in source
    assert source.index("write_phase2g_dependency_graph.py") < source.index('scontrol release "$joined"')
    assert source.index('>"$GATE_ROOT/release-attempted"') < source.index('scontrol release "$joined"')
    assert '[[ -e "$GATE_ROOT/release-attempted" ]]' in source
    assert "--validate-only" in source
    assert '--registry "$REGISTRY" --ledger "$LEDGER" --readiness "$READINESS"' in source
    assert 'VALIDATION_STATE_ROOT="${JEPA4D_VALIDATION_STATE_ROOT:-$ROOT/outputs/validation-state}"' in source
    assert 'export JEPA4D_VALIDATION_STATE_ROOT="$VALIDATION_STATE_ROOT"' in source
    assert "JEPA4D_VALIDATION_STATE_ROOT=$VALIDATION_STATE_ROOT" in source
    assert 'execution_branch="phase2g-exec-${SHORT}-${NONCE}"' in source
    assert 'anchor_branch="phase2g-pushed-${SHORT}-${NONCE}"' in source
    assert 'branch --set-upstream-to="$anchor_branch"' in source
    assert "JEPA4D_LOG_ROOT=$LOG_ROOT" in source
    assert 'export PYTHONPATH="$ROOT"' in source
    assert "PYTHONPATH=$ROOT" in source
    assert "JEPA4D_EXECUTION_WORKTREE=1" in source
    assert "submitted=false" in source
    assert "ALL," not in source
    assert "JEPA4D_DIODE" not in source
    assert "DIODE_ARCHIVE" not in source
    expected_dependencies = (
        'O="$(submit "$name" "$T"',
        'C="$(submit "$name" "$T"',
        'Q="$(submit "$name" "$O:$C"',
        'H="$(submit "$name" "$Q"',
        'HG="$(submit "$name" "$H"',
        'F="$(submit "$name" "$HG"',
        'V="$(submit "$name" "$F"',
        'S="$(submit "$name" "$V"',
        'G="$(submit "$name" "$S"',
        'Z="$(submit "$name" "$G"',
    )
    assert all(value in source for value in expected_dependencies)


def test_dispatch_removes_full_cache_root_and_passes_only_authorized_views() -> None:
    path = ROOT / "slurm" / "phase2g_array_dispatch.sbatch"
    source = path.read_text(encoding="utf-8")
    subprocess.run(("bash", "-n", str(path)), check=True)
    assert "TASK_ID >= 0 && TASK_ID < 48" in source
    assert "arm_index=$((TASK_ID / 12))" in source
    assert "rotation_index=$((within_arm / 3))" in source
    assert "rotations/$JEPA4D_ROTATION" in source
    assert source.count("unset JEPA4D_CACHE_ROOT CACHE_ROOT") == 3
    assert 'JEPA4D_INPUT_SHARD="$CACHE_ROOT/shards/$heldout/input.pt"' in source
    assert 'JEPA4D_FEATURE_SHARD="$CACHE_ROOT/shards/$heldout/feature.pt"' in source
    assert 'JEPA4D_TARGET_SHARD="$CACHE_ROOT/shards/$heldout/target.pt"' in source
    assert 'exec bash "$ROOT/slurm/phase2g_tune.sbatch"' in source
    assert 'exec bash "$ROOT/slurm/phase2g_train.sbatch"' in source
    assert 'exec bash "$ROOT/slurm/phase2g_evaluate.sbatch"' in source
    assert "eval " not in source


def test_core_runner_cli_contract_is_exact() -> None:
    expected = {
        "phase2g_cache.sbatch": (
            "materialize_phase2g_sun.py",
            "build_phase2g_data_cache.py",
            "--sun-root",
            "--materialization-receipt",
            "--vjepa-checkpoint",
            "--vjepa-implementation",
            "--provenance",
            "--output",
        ),
        "phase2g_audit.sbatch": (
            "audit_phase2g_formal.py",
            "--cache-root",
            "--materialization-root",
            "--provenance",
            "--output",
        ),
        "phase2g_tune.sbatch": (
            "run_phase2g_tuning.py",
            "--arm",
            "--rotation",
            "--learning-rate",
            "--cache-root",
            "--provenance",
            "--output",
        ),
        "phase2g_train.sbatch": (
            "run_phase2g_formal_training.py",
            "--arm",
            "--rotation",
            "--seed",
            "--cache-root",
            "--lr-selection",
            "--provenance",
            "--output",
        ),
        "phase2g_evaluate.sbatch": (
            "evaluate_phase2g_heldout.py",
            "--input-shard",
            "--feature-shard",
            "--target-shard",
            "--training-receipt",
            "--provenance",
            "--output",
        ),
        "phase2g_lr_select.sbatch": ("select_phase2g_learning_rates.py", "--receipt"),
        "phase2g_select.sbatch": ("select_phase2g_survivor.py", "--evaluation-receipt"),
    }
    for filename, markers in expected.items():
        source = (ROOT / "slurm" / filename).read_text(encoding="utf-8")
        assert all(marker in source for marker in markers)


def test_graph_parser_binds_submission_and_runtime_entrypoints() -> None:
    value = _parse_job(
        "H-M0-R0-L0|12345_0|p2gq-H-M0-R0-L0-deadbeef|Q|"
        "slurm/phase2g_array_dispatch.sbatch|slurm/phase2g_tune.sbatch|outputs/tune.json",
        ROOT,
    )
    assert value["job_id"] == "12345_0"
    assert value["parents"] == ["Q"]
    assert value["submission_sbatch"]["relative_path"] == "slurm/phase2g_array_dispatch.sbatch"
    assert value["entrypoint"]["relative_path"] == "slurm/phase2g_tune.sbatch"
