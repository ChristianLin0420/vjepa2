#!/usr/bin/env bash
#SBATCH --job-name=jepa4d-p2e-final
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
[[ -n "${SLURM_JOB_ID:-}" ]] || jepa4d_die "Phase 2e final evaluation must run through sbatch"
jepa4d_activate_python "$REPO_ROOT"
jepa4d_init_log_dir "$REPO_ROOT" "phase2e-final"
jepa4d_install_traps
TEST_REPORT="${JEPA4D_TEST_REPORT:-$REPO_ROOT/outputs/phase2d-gates/tests.json}"
CACHE_ROOT="${JEPA4D_CACHE_ROOT:-$REPO_ROOT/outputs/jepa4d_phase2e/sunrgbd-cache}"
MANIFEST="${JEPA4D_SUNRGBD_MANIFEST:-$REPO_ROOT/jepa4d/config/benchmarks/manifests/sun_rgbd_phase2e_sensor_blocked_v1.yaml}"
FORMAL_ROOT="${JEPA4D_FORMAL_ROOT:-$REPO_ROOT/outputs/jepa4d_phase2e/formal}"
OUTPUT="${JEPA4D_OUTPUT:-$REPO_ROOT/outputs/jepa4d_phase2e/final}"
: "${JEPA4D_DEPENDENCY_GRAPH:?export the exact submitted Phase 2d/2e job graph before final evaluation}"
jepa4d_require_file "$TEST_REPORT" "passing Phase 2d/2e test receipt"
jepa4d_require_file "$MANIFEST" "frozen SUN RGB-D manifest"
for shard in 0 1 2 3; do jepa4d_require_dir "$FORMAL_ROOT/shard-$shard" "formal Phase 2e shard-$shard"; done
if [[ -d "$OUTPUT" && -n "$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  jepa4d_die "final evaluation output is not empty: $OUTPUT"
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
variants=(
  "monolithic_final,factorized_bias"
  "factorized_vjepa,factorized_rgb"
  "factorized_vjepa_rgb,factorized_vjepa_k"
  "factorized_full,factorized_full_teacher"
)
for shard in 0 1 2 3; do
  jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/slurm/validate_phase2e_shard.py" \
    --output "$FORMAL_ROOT/shard-$shard" --expected-variants "${variants[$shard]}" \
    --expected-epochs 60 --expected-feature-cache "$CACHE_ROOT/train_validation_cache.pt"
done
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/scripts/check_cuda.py" \
  --device "${JEPA4D_CUDA_DEVICE_INDEX:-0}" --stress-seconds 10 --matrix-size 4096 --allocation-mib 1024 \
  --json-output "$JEPA4D_JOB_LOG_DIR/cuda-health.json"
command=(
  "$JEPA4D_PYTHON" "$REPO_ROOT/scripts/evaluate_phase2e_final.py"
  --train-validation-cache "$CACHE_ROOT/train_validation_cache.pt"
  --test-cache "$CACHE_ROOT/test_cache.pt"
  --feature-cache-receipt "$CACHE_ROOT/feature_cache_receipt.json"
  --test-receipt "$TEST_REPORT"
  --shard-dir "$FORMAL_ROOT/shard-0" --shard-dir "$FORMAL_ROOT/shard-1"
  --shard-dir "$FORMAL_ROOT/shard-2" --shard-dir "$FORMAL_ROOT/shard-3"
  --output "$OUTPUT" --device "cuda:${JEPA4D_CUDA_DEVICE_INDEX:-0}" --batch-size 8
  --latency-warmup 20 --latency-iterations 100 --latency-repetitions 5
  --wandb-project "${JEPA4D_WANDB_PROJECT:-jepa4d-worldmodel}"
  --run-name "phase2e-final-kv2-${SLURM_JOB_ID}"
)
if [[ -n "${JEPA4D_WANDB_ENTITY:-}" ]]; then command+=(--wandb-entity "$JEPA4D_WANDB_ENTITY"); fi
cd "$REPO_ROOT"
jepa4d_run_step "${command[@]}"
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/slurm/validate_phase2e_final.py" --output "$OUTPUT" | \
  tee "$OUTPUT/postflight.json"
printf 'pass\n' >"$JEPA4D_JOB_LOG_DIR/SUCCESS"
