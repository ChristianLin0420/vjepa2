#!/usr/bin/env bash
#SBATCH --job-name=jepa4d-p2e-train
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=polar4,polar3,polar,batch_block1,grizzly,batch_block2,batch_block3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
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
[[ -n "${SLURM_JOB_ID:-}" ]] || jepa4d_die "Phase 2e training must run through sbatch"
jepa4d_activate_python "$REPO_ROOT"
jepa4d_init_log_dir "$REPO_ROOT" "phase2e-train"
jepa4d_install_traps
TEST_REPORT="${JEPA4D_TEST_REPORT:-$REPO_ROOT/outputs/phase2d-gates/tests.json}"
CACHE_ROOT="${JEPA4D_CACHE_ROOT:-$REPO_ROOT/outputs/jepa4d_phase2e/sunrgbd-cache}"
CACHE="$CACHE_ROOT/train_validation_cache.pt"
MANIFEST="${JEPA4D_SUNRGBD_MANIFEST:-$REPO_ROOT/jepa4d/config/benchmarks/manifests/sun_rgbd_phase2e_sensor_blocked_v1.yaml}"
MODE="${JEPA4D_TRAIN_MODE:-formal}"
SHARD="${JEPA4D_SHARD_ID:?export JEPA4D_SHARD_ID before submitting}"
if [[ "$MODE" == "formal" ]]; then
  EPOCHS="${JEPA4D_EPOCHS:-60}"
  [[ "$EPOCHS" == "60" ]] || jepa4d_die "formal Phase 2e shards are locked to 60 epochs"
  case "$SHARD" in
    0) VARIANTS="monolithic_final,factorized_bias" ;;
    1) VARIANTS="factorized_vjepa,factorized_rgb" ;;
    2) VARIANTS="factorized_vjepa_rgb,factorized_vjepa_k" ;;
    3) VARIANTS="factorized_full,factorized_full_teacher" ;;
    *) jepa4d_die "formal JEPA4D_SHARD_ID must be exactly one of 0, 1, 2, or 3" ;;
  esac
  PILOT_OUTPUT="${JEPA4D_PILOT_OUTPUT:-$REPO_ROOT/outputs/jepa4d_phase2e/pilot/shard-pilot}"
elif [[ "$MODE" == "pilot" ]]; then
  EPOCHS="${JEPA4D_EPOCHS:-2}"
  [[ "$EPOCHS" == "2" ]] || jepa4d_die "Phase 2e pilot is locked to 2 epochs"
  [[ "$SHARD" == "pilot" ]] || jepa4d_die "pilot JEPA4D_SHARD_ID must be exactly 'pilot'"
  VARIANTS="monolithic_final,factorized_full_teacher"
else
  jepa4d_die "JEPA4D_TRAIN_MODE must be pilot or formal"
fi
if [[ -n "${JEPA4D_VARIANTS:-}" && "$JEPA4D_VARIANTS" != "$VARIANTS" ]]; then
  jepa4d_die "JEPA4D_VARIANTS does not match the frozen $MODE shard-$SHARD map: $VARIANTS"
fi
OUTPUT_ROOT="${JEPA4D_OUTPUT_ROOT:-$REPO_ROOT/outputs/jepa4d_phase2e/$MODE}"
OUTPUT="$OUTPUT_ROOT/shard-$SHARD"
jepa4d_require_file "$TEST_REPORT" "passing Phase 2d/2e test receipt"
jepa4d_require_file "$MANIFEST" "frozen SUN RGB-D sensor-blocked manifest"
jepa4d_require_file "$CACHE" "train/validation feature cache"
jepa4d_require_file "$CACHE_ROOT/feature_cache_receipt.json" "feature-cache receipt"
jepa4d_require_file "$CACHE_ROOT/wandb_receipt.json" "feature-cache W&B receipt"
if [[ -d "$OUTPUT" && -n "$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  jepa4d_die "training output is not empty: $OUTPUT"
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
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/slurm/validate_phase2e_cache.py" \
  --cache-root "$CACHE_ROOT" --manifest "$MANIFEST"
if [[ "$MODE" == "formal" ]]; then
  jepa4d_require_dir "$PILOT_OUTPUT" "successful Phase 2e pilot output"
  jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/slurm/validate_phase2e_shard.py" \
    --output "$PILOT_OUTPUT" \
    --expected-variants "monolithic_final,factorized_full_teacher" \
    --expected-epochs 2 \
    --expected-feature-cache "$CACHE"
fi
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/scripts/check_cuda.py" \
  --device "${JEPA4D_CUDA_DEVICE_INDEX:-0}" --stress-seconds 10 --matrix-size 4096 --allocation-mib 1024 \
  --json-output "$JEPA4D_JOB_LOG_DIR/cuda-health.json"
command=(
  "$JEPA4D_PYTHON" "$REPO_ROOT/scripts/run_phase2e_factorized_shard.py"
  --cache "$CACHE" --output "$OUTPUT" --variants "$VARIANTS" --seeds 0,1,2
  --epochs "$EPOCHS" --batch-size 8 --hidden-dim 64 --device "cuda:${JEPA4D_CUDA_DEVICE_INDEX:-0}"
  --wandb-project "${JEPA4D_WANDB_PROJECT:-jepa4d-worldmodel}"
  --run-name "phase2e-${MODE}-shard-${SHARD}-${SLURM_JOB_ID}"
)
if [[ -n "${JEPA4D_WANDB_ENTITY:-}" ]]; then command+=(--wandb-entity "$JEPA4D_WANDB_ENTITY"); fi
cd "$REPO_ROOT"
jepa4d_run_step "${command[@]}"
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/slurm/validate_phase2e_shard.py" \
  --output "$OUTPUT" --expected-variants "$VARIANTS" --expected-epochs "$EPOCHS" \
  --expected-feature-cache "$CACHE"
printf 'pass\n' >"$JEPA4D_JOB_LOG_DIR/SUCCESS"
