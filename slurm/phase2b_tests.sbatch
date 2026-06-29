#!/usr/bin/env bash
#SBATCH --job-name=jepa4d-p2b-tests
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
jepa4d_activate_python "$REPO_ROOT"
jepa4d_init_log_dir "$REPO_ROOT" "phase2b-tests"
jepa4d_install_traps
jepa4d_capture_environment "$REPO_ROOT" "$JEPA4D_JOB_LOG_DIR"
jepa4d_start_gpu_monitor "$JEPA4D_JOB_LOG_DIR/gpu-telemetry.csv"
TEST_REPORT="${JEPA4D_TEST_REPORT:-$REPO_ROOT/outputs/phase2b-gates/tests.json}"
mkdir -p "$(dirname "$TEST_REPORT")"
rm -f "$TEST_REPORT"

printf '\n[%s] shell and Python syntax validation\n' "$(date --iso-8601=seconds)"
jepa4d_run_step bash -n \
  "$REPO_ROOT/slurm/lib.sh" \
  "$REPO_ROOT/slurm/prepare_phase2b_login.sh" \
  "$REPO_ROOT/slurm/phase2b_setup.sbatch" \
  "$REPO_ROOT/slurm/phase2b_tests.sbatch" \
  "$REPO_ROOT/slurm/phase2b_preflight.sbatch" \
  "$REPO_ROOT/slurm/phase2b_train.sbatch"
jepa4d_run_step "$JEPA4D_PYTHON" -m compileall -q \
  "$REPO_ROOT/jepa4d" "$REPO_ROOT/scripts" "$REPO_ROOT/slurm"

if "$JEPA4D_PYTHON" -c 'import ruff' >/dev/null 2>&1; then
  printf '\n[%s] Ruff\n' "$(date --iso-8601=seconds)"
  jepa4d_run_step "$JEPA4D_PYTHON" -m ruff format --check \
    "$REPO_ROOT/jepa4d" "$REPO_ROOT/scripts" "$REPO_ROOT/slurm"
  jepa4d_run_step "$JEPA4D_PYTHON" -m ruff check \
    "$REPO_ROOT/jepa4d" "$REPO_ROOT/scripts" "$REPO_ROOT/slurm"
elif [[ "${JEPA4D_REQUIRE_STATIC_TOOLS:-0}" == "1" ]]; then
  jepa4d_die "ruff is required but is not installed"
else
  printf 'ruff unavailable; syntax checks still ran\n'
fi

if "$JEPA4D_PYTHON" -c 'import mypy' >/dev/null 2>&1; then
  printf '\n[%s] mypy\n' "$(date --iso-8601=seconds)"
  jepa4d_run_step "$JEPA4D_PYTHON" -m mypy \
    "$REPO_ROOT/jepa4d" \
    "$REPO_ROOT/scripts/check_cuda.py" "$REPO_ROOT/scripts/run_phase2b_geometry_distillation.py" \
    "$REPO_ROOT/slurm/download_phase2b_assets.py" "$REPO_ROOT/slurm/phase2b_gate.py" \
    "$REPO_ROOT/slurm/phase2b_preflight.py" "$REPO_ROOT/slurm/validate_phase2b_preflight.py" \
    "$REPO_ROOT/slurm/validate_phase2b_output.py" "$REPO_ROOT/slurm/write_phase2b_test_receipt.py"
elif [[ "${JEPA4D_REQUIRE_STATIC_TOOLS:-0}" == "1" ]]; then
  jepa4d_die "mypy is required but is not installed"
else
  printf 'mypy unavailable; syntax checks still ran\n'
fi

printf '\n[%s] JEPA-4D unit and contract tests\n' "$(date --iso-8601=seconds)"
jepa4d_run_step "$JEPA4D_PYTHON" -m pytest "$REPO_ROOT/jepa4d/tests" -ra -q

printf '\n[%s] allocated-GPU smoke and sustained compute\n' "$(date --iso-8601=seconds)"
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/scripts/check_cuda.py" \
  --device "${JEPA4D_CUDA_DEVICE_INDEX:-0}" \
  --stress-seconds "${JEPA4D_TEST_CUDA_STRESS_SECONDS:-15}" \
  --matrix-size "${JEPA4D_CUDA_MATRIX_SIZE:-4096}" \
  --allocation-mib "${JEPA4D_TEST_CUDA_ALLOCATION_MIB:-1024}" \
  --json-output "$JEPA4D_JOB_LOG_DIR/cuda-health.json"

printf '\n[%s] durable passing-test receipt\n' "$(date --iso-8601=seconds)"
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/slurm/write_phase2b_test_receipt.py" \
  --repo-root "$REPO_ROOT" \
  --cuda-report "$JEPA4D_JOB_LOG_DIR/cuda-health.json" \
  --output "$TEST_REPORT"

printf 'pass\n' >"$JEPA4D_JOB_LOG_DIR/SUCCESS"
printf '[%s] tests passed; receipt=%s\n' "$(date --iso-8601=seconds)" "$TEST_REPORT"
