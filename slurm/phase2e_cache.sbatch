#!/usr/bin/env bash
#SBATCH --job-name=jepa4d-p2e-cache
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=polar4,polar3,polar,batch_block1,grizzly,batch_block2,batch_block3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=04:00:00
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
[[ -n "${SLURM_JOB_ID:-}" ]] || jepa4d_die "Phase 2e cache build must run through sbatch"
jepa4d_activate_python "$REPO_ROOT"
jepa4d_init_log_dir "$REPO_ROOT" "phase2e-cache"
jepa4d_install_traps
jepa4d_asset_paths "$REPO_ROOT"
jepa4d_validate_assets
TEST_REPORT="${JEPA4D_TEST_REPORT:-$REPO_ROOT/outputs/phase2d-gates/tests.json}"
DATASET_ROOT="${JEPA4D_SUNRGBD_ROOT:?export JEPA4D_SUNRGBD_ROOT before submitting}"
MANIFEST="${JEPA4D_SUNRGBD_MANIFEST:-$REPO_ROOT/jepa4d/config/benchmarks/manifests/sun_rgbd_phase2e_sensor_blocked_v1.yaml}"
OUTPUT="${JEPA4D_OUTPUT:-$REPO_ROOT/outputs/jepa4d_phase2e/sunrgbd-cache}"
jepa4d_require_file "$TEST_REPORT" "passing Phase 2d/2e test receipt"
jepa4d_require_dir "$DATASET_ROOT" "SUN RGB-D root"
jepa4d_require_file "$MANIFEST" "SUN RGB-D sensor-blocked manifest"
if [[ -d "$OUTPUT" && -n "$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  jepa4d_die "cache output is not empty: $OUTPUT"
fi
mkdir -p "$OUTPUT"
export WANDB_MODE=online
export WANDB_DIR="${WANDB_DIR:-$JEPA4D_JOB_LOG_DIR/wandb}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-180}"
mkdir -p "$WANDB_DIR"
jepa4d_capture_environment "$REPO_ROOT" "$JEPA4D_JOB_LOG_DIR"
jepa4d_start_gpu_monitor "$JEPA4D_JOB_LOG_DIR/gpu-telemetry.csv"
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/slurm/validate_phase2d_test_receipt.py" \
  --repo-root "$REPO_ROOT" --receipt "$TEST_REPORT"
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/scripts/check_cuda.py" \
  --device "${JEPA4D_CUDA_DEVICE_INDEX:-0}" --stress-seconds 20 --matrix-size 4096 --allocation-mib 2048 \
  --json-output "$JEPA4D_JOB_LOG_DIR/cuda-health.json"
command=(
  "$JEPA4D_PYTHON" "$REPO_ROOT/scripts/build_phase2e_sunrgbd_feature_cache.py"
  --dataset-root "$DATASET_ROOT" --manifest "$MANIFEST"
  --vjepa-checkpoint "$JEPA4D_VJEPA_CHECKPOINT" --vjepa-implementation "$JEPA4D_VJEPA_IMPLEMENTATION"
  --vggt-checkpoint "$JEPA4D_VGGT_CHECKPOINT" --output "$OUTPUT"
  --device "cuda:${JEPA4D_CUDA_DEVICE_INDEX:-0}" --vjepa-chunk-size 8 --vggt-chunk-size 2
  --wandb-project "${JEPA4D_WANDB_PROJECT:-jepa4d-worldmodel}"
  --run-name "phase2e-sunrgbd-cache-${SLURM_JOB_ID}"
)
if [[ -n "${JEPA4D_WANDB_ENTITY:-}" ]]; then command+=(--wandb-entity "$JEPA4D_WANDB_ENTITY"); fi
cd "$REPO_ROOT"
jepa4d_run_step "${command[@]}"
for required in train_validation_cache.pt test_cache.pt feature_normalization.pt feature_cache_receipt.json \
  feature_cache_report.html wandb_receipt.json; do
  jepa4d_require_file "$OUTPUT/$required" "$required"
done
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/slurm/validate_phase2e_cache.py" \
  --cache-root "$OUTPUT" --manifest "$MANIFEST"
if grep -Eiq '<script[^>]+src=' "$OUTPUT/feature_cache_report.html"; then
  jepa4d_die "cache HTML contains an external script dependency"
fi
printf 'pass\n' >"$JEPA4D_JOB_LOG_DIR/SUCCESS"
