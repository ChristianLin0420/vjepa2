#!/usr/bin/env bash
#SBATCH --job-name=jepa4d-p2c-tests
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=polar4,polar3,polar,batch_block1,grizzly,batch_block2,batch_block3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@120

set -Eeuo pipefail
IFS=$'\n\t'
umask 027

bootstrap="${JEPA4D_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "$bootstrap/slurm/lib.sh" && -f "$bootstrap/../slurm/lib.sh" ]]; then
  bootstrap="$bootstrap/.."
fi
[[ -f "$bootstrap/slurm/lib.sh" ]] || {
  printf 'ERROR: submit from the repository root or export JEPA4D_REPO_ROOT\n' >&2
  exit 2
}
# shellcheck disable=SC1091
source "$bootstrap/slurm/lib.sh"

REPO_ROOT="$(jepa4d_find_repo_root "$bootstrap")"
[[ -n "${SLURM_JOB_ID:-}" ]] || jepa4d_die "Phase 2c tests must run through sbatch"
jepa4d_activate_python "$REPO_ROOT"
jepa4d_init_log_dir "$REPO_ROOT" "phase2c-tests"
jepa4d_install_traps
jepa4d_capture_environment "$REPO_ROOT" "$JEPA4D_JOB_LOG_DIR"
jepa4d_start_gpu_monitor "$JEPA4D_JOB_LOG_DIR/gpu-telemetry.csv"
TEST_REPORT="${JEPA4D_TEST_REPORT:-$REPO_ROOT/outputs/phase2c-gates/tests.json}"
mkdir -p "$(dirname "$TEST_REPORT")"
rm -f "$TEST_REPORT"

printf '\n[%s] shell and Python syntax validation\n' "$(date --iso-8601=seconds)"
jepa4d_run_step bash -n \
  "$REPO_ROOT/slurm/prepare_phase2c_login.sh" \
  "$REPO_ROOT/slurm/phase2c_tests.sbatch" \
  "$REPO_ROOT/slurm/phase2c_preflight.sbatch" \
  "$REPO_ROOT/slurm/phase2c_train.sbatch"
jepa4d_run_step "$JEPA4D_PYTHON" -m compileall -q \
  "$REPO_ROOT/jepa4d" "$REPO_ROOT/scripts" "$REPO_ROOT/slurm"

printf '\n[%s] Ruff\n' "$(date --iso-8601=seconds)"
jepa4d_run_step "$JEPA4D_PYTHON" -m ruff format --check \
  "$REPO_ROOT/jepa4d" "$REPO_ROOT/scripts" "$REPO_ROOT/slurm"
jepa4d_run_step "$JEPA4D_PYTHON" -m ruff check \
  "$REPO_ROOT/jepa4d" "$REPO_ROOT/scripts" "$REPO_ROOT/slurm"

printf '\n[%s] mypy\n' "$(date --iso-8601=seconds)"
jepa4d_run_step "$JEPA4D_PYTHON" -m mypy \
  "$REPO_ROOT/jepa4d" \
  "$REPO_ROOT/scripts/check_cuda.py" \
  "$REPO_ROOT/scripts/run_phase2b_geometry_distillation.py" \
  "$REPO_ROOT/slurm/phase2c_gate.py" \
  "$REPO_ROOT/slurm/download_phase2c_assets.py" \
  "$REPO_ROOT/slurm/write_phase2c_test_receipt.py" \
  "$REPO_ROOT/slurm/phase2c_preflight.py" \
  "$REPO_ROOT/slurm/validate_phase2c_preflight.py" \
  "$REPO_ROOT/slurm/validate_phase2c_output.py"

printf '\n[%s] complete JEPA-4D tests\n' "$(date --iso-8601=seconds)"
jepa4d_run_step "$JEPA4D_PYTHON" -m pytest "$REPO_ROOT/jepa4d/tests" -ra -q

printf '\n[%s] allocated-GPU smoke and sustained compute\n' "$(date --iso-8601=seconds)"
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/scripts/check_cuda.py" \
  --device "${JEPA4D_CUDA_DEVICE_INDEX:-0}" \
  --stress-seconds "${JEPA4D_TEST_CUDA_STRESS_SECONDS:-15}" \
  --matrix-size "${JEPA4D_CUDA_MATRIX_SIZE:-4096}" \
  --allocation-mib "${JEPA4D_TEST_CUDA_ALLOCATION_MIB:-1024}" \
  --json-output "$JEPA4D_JOB_LOG_DIR/cuda-health.json"

printf '\n[%s] durable passing-test receipt\n' "$(date --iso-8601=seconds)"
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/slurm/write_phase2c_test_receipt.py" \
  --repo-root "$REPO_ROOT" \
  --cuda-report "$JEPA4D_JOB_LOG_DIR/cuda-health.json" \
  --output "$TEST_REPORT"

printf 'pass\n' >"$JEPA4D_JOB_LOG_DIR/SUCCESS"
printf '[%s] Phase 2c tests passed; receipt=%s\n' "$(date --iso-8601=seconds)" "$TEST_REPORT"
