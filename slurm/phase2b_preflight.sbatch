#!/usr/bin/env bash
#SBATCH --job-name=jepa4d-p2b-preflight
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
[[ -n "${SLURM_JOB_ID:-}" ]] || jepa4d_die "Phase 2b preflight must run through sbatch"
jepa4d_activate_python "$REPO_ROOT"
jepa4d_init_log_dir "$REPO_ROOT" "phase2b-preflight"
jepa4d_install_traps
jepa4d_asset_paths "$REPO_ROOT"
jepa4d_validate_assets

DATASET_ROOT="${JEPA4D_DATASET_ROOT:?export JEPA4D_DATASET_ROOT before submitting}"
TUM_ARCHIVE="${JEPA4D_TUM_ARCHIVE:?export JEPA4D_TUM_ARCHIVE before submitting}"
MANIFEST="${JEPA4D_MANIFEST:-$REPO_ROOT/jepa4d/config/benchmarks/manifests/tum_rgbd_phase2b_v1.yaml}"
jepa4d_require_dir "$DATASET_ROOT" "TUM dataset root"
jepa4d_require_file "$TUM_ARCHIVE" "TUM archive"
jepa4d_require_file "$MANIFEST" "Phase 2b manifest"
DATASET_ROOT="$(jepa4d_realpath "$DATASET_ROOT")"
TUM_ARCHIVE="$(jepa4d_realpath "$TUM_ARCHIVE")"
MANIFEST="$(jepa4d_realpath "$MANIFEST")"

TEST_REPORT="${JEPA4D_TEST_REPORT:-$REPO_ROOT/outputs/phase2b-gates/tests.json}"
PREFLIGHT_REPORT="${JEPA4D_PREFLIGHT_REPORT:-$REPO_ROOT/outputs/phase2b-gates/preflight.json}"
CUDA_REPORT="$JEPA4D_JOB_LOG_DIR/cuda-health.json"
jepa4d_require_file "$TEST_REPORT" "passing Phase 2b Slurm test receipt"
mkdir -p "$(dirname "$PREFLIGHT_REPORT")"
rm -f "$PREFLIGHT_REPORT"
export WANDB_MODE=online
export WANDB_DIR="${WANDB_DIR:-$JEPA4D_JOB_LOG_DIR/wandb}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-180}"
mkdir -p "$WANDB_DIR"

jepa4d_capture_environment "$REPO_ROOT" "$JEPA4D_JOB_LOG_DIR"
jepa4d_start_gpu_monitor "$JEPA4D_JOB_LOG_DIR/gpu-telemetry.csv"

printf '\n[%s] sustained CUDA health check\n' "$(date --iso-8601=seconds)"
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/scripts/check_cuda.py" \
  --device "${JEPA4D_CUDA_DEVICE_INDEX:-0}" \
  --stress-seconds "${JEPA4D_CUDA_STRESS_SECONDS:-45}" \
  --matrix-size "${JEPA4D_CUDA_MATRIX_SIZE:-4096}" \
  --allocation-mib "${JEPA4D_CUDA_ALLOCATION_MIB:-2048}" \
  --json-output "$CUDA_REPORT"

printf '\n[%s] dataset, asset, and real-model smoke tests\n' "$(date --iso-8601=seconds)"
preflight_command=(
  "$JEPA4D_PYTHON" "$REPO_ROOT/slurm/phase2b_preflight.py"
  --dataset-root "$DATASET_ROOT"
  --archive "$TUM_ARCHIVE"
  --manifest "$MANIFEST"
  --vjepa-checkpoint "$JEPA4D_VJEPA_CHECKPOINT"
  --vjepa-implementation "$JEPA4D_VJEPA_IMPLEMENTATION"
  --vggt-checkpoint "$JEPA4D_VGGT_CHECKPOINT"
  --repo-root "$REPO_ROOT"
  --test-report "$TEST_REPORT"
  --device "cuda:${JEPA4D_CUDA_DEVICE_INDEX:-0}"
  --wandb-project "${JEPA4D_WANDB_PROJECT:-jepa4d-worldmodel}"
  --wandb-run-name "phase2b-preflight-${SLURM_JOB_ID:-manual}"
  --output "$PREFLIGHT_REPORT"
)
if [[ -n "${JEPA4D_WANDB_ENTITY:-}" ]]; then
  preflight_command+=(--wandb-entity "$JEPA4D_WANDB_ENTITY")
fi
jepa4d_run_step "${preflight_command[@]}"

printf 'pass\n' >"$JEPA4D_JOB_LOG_DIR/SUCCESS"
printf '[%s] preflight passed; report=%s\n' "$(date --iso-8601=seconds)" "$PREFLIGHT_REPORT"
