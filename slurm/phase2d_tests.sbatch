#!/usr/bin/env bash
#SBATCH --job-name=jepa4d-p2d-tests
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=polar4,polar3,polar,batch_block1,grizzly,batch_block2,batch_block3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@120

set -Eeuo pipefail
IFS=$'\n\t'
umask 027
bootstrap="${JEPA4D_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "$bootstrap/slurm/lib.sh" && -f "$bootstrap/../slurm/lib.sh" ]]; then bootstrap="$bootstrap/.."; fi
[[ -f "$bootstrap/slurm/lib.sh" ]] || exit 2
# shellcheck disable=SC1091
source "$bootstrap/slurm/lib.sh"
REPO_ROOT="$(jepa4d_find_repo_root "$bootstrap")"
[[ -n "${SLURM_JOB_ID:-}" ]] || jepa4d_die "Phase 2d tests must run through sbatch"
jepa4d_activate_python "$REPO_ROOT"
jepa4d_init_log_dir "$REPO_ROOT" "phase2d-tests"
jepa4d_install_traps
RECEIPT="${JEPA4D_TEST_REPORT:-$REPO_ROOT/outputs/phase2d-gates/tests.json}"
mkdir -p "$(dirname "$RECEIPT")"
rm -f "$RECEIPT"
jepa4d_capture_environment "$REPO_ROOT" "$JEPA4D_JOB_LOG_DIR"
jepa4d_start_gpu_monitor "$JEPA4D_JOB_LOG_DIR/gpu-telemetry.csv"

cd "$REPO_ROOT"
jepa4d_run_step bash -n "$REPO_ROOT"/slurm/phase2d_*.sbatch "$REPO_ROOT"/slurm/phase2e_*.sbatch
jepa4d_run_step "$JEPA4D_PYTHON" -m compileall -q "$REPO_ROOT/jepa4d" "$REPO_ROOT/scripts" "$REPO_ROOT/slurm"
jepa4d_run_step "$JEPA4D_PYTHON" -m ruff format --check "$REPO_ROOT/jepa4d" "$REPO_ROOT/scripts" "$REPO_ROOT/slurm"
jepa4d_run_step "$JEPA4D_PYTHON" -m ruff check "$REPO_ROOT/jepa4d" "$REPO_ROOT/scripts" "$REPO_ROOT/slurm"
jepa4d_run_step "$JEPA4D_PYTHON" -m mypy "$REPO_ROOT/jepa4d" \
  "$REPO_ROOT/scripts/run_phase2d_fusion_attribution.py" \
  "$REPO_ROOT/scripts/run_phase2d_calibration_audit.py" \
  "$REPO_ROOT/scripts/run_phase2d_latency_confirmation.py" \
  "$REPO_ROOT/scripts/aggregate_phase2d_latency.py" \
  "$REPO_ROOT/scripts/aggregate_phase2d_diagnostics.py" \
  "$REPO_ROOT/slurm/validate_phase2d_diagnostics.py" \
  "$REPO_ROOT/scripts/build_phase2e_sunrgbd_feature_cache.py" \
  "$REPO_ROOT/scripts/run_phase2e_factorized_shard.py" \
  "$REPO_ROOT/scripts/evaluate_phase2e_final.py" \
  "$REPO_ROOT/scripts/write_phase2_dependency_graphs.py" \
  "$REPO_ROOT/slurm/validate_phase2e_cache.py" \
  "$REPO_ROOT/slurm/validate_phase2e_shard.py" \
  "$REPO_ROOT/slurm/validate_phase2e_final.py"
jepa4d_run_step "$JEPA4D_PYTHON" -m pytest "$REPO_ROOT/jepa4d/tests" -ra -q
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/scripts/check_cuda.py" \
  --device "${JEPA4D_CUDA_DEVICE_INDEX:-0}" --stress-seconds 20 --matrix-size 4096 --allocation-mib 1024 \
  --json-output "$JEPA4D_JOB_LOG_DIR/cuda-health.json"
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/slurm/write_phase2d_test_receipt.py" \
  --repo-root "$REPO_ROOT" --cuda-report "$JEPA4D_JOB_LOG_DIR/cuda-health.json" --output "$RECEIPT"
printf 'pass\n' >"$JEPA4D_JOB_LOG_DIR/SUCCESS"
