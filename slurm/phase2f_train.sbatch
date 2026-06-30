#!/usr/bin/env bash
#SBATCH --job-name=j4d-p2f-train
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=polar4,polar3,polar,batch_block1,grizzly,batch_block2,batch_block3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --signal=B:USR1@120

set -Eeuo pipefail
IFS=$'\n\t'
umask 027
ROOT="${JEPA4D_REPO_ROOT:?}"
source "$ROOT/slurm/lib.sh"
REPO_ROOT="$(jepa4d_find_repo_root "$ROOT")"
jepa4d_activate_python "$REPO_ROOT"
jepa4d_init_log_dir "$REPO_ROOT" "phase2f-train"
jepa4d_install_traps
jepa4d_capture_environment "$REPO_ROOT" "$JEPA4D_JOB_LOG_DIR"
jepa4d_start_gpu_monitor "$JEPA4D_JOB_LOG_DIR/gpu-telemetry.csv"
export WANDB_MODE=online PYTHONUTF8=1
MODE="${JEPA4D_TRAIN_MODE:?}"
ARM="${JEPA4D_ARM:?}"
ROTATION="${JEPA4D_ROTATION:?}"
SEED="${JEPA4D_SEED:?}"
LABEL="${JEPA4D_JOB_LABEL:?}"
OUT="${JEPA4D_STAGE_OUTPUT:?}"
CACHE="${JEPA4D_CACHE_ROOT:?}"
PROVENANCE="$JEPA4D_JOB_LOG_DIR/execution_provenance.json"
if [[ "$MODE" == pilot ]]; then
  PARENT_LABEL=LA
  PARENT_RECEIPT="${JEPA4D_LATENCY_GATE:?}"
else
  PARENT_LABEL=PG
  PARENT_RECEIPT="${JEPA4D_PILOT_GATE:?}"
fi
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/slurm/phase2f_contract.py" emit-provenance \
  --graph "${JEPA4D_GRAPH:?}" --label "$LABEL" --test-receipt "${JEPA4D_TEST_RECEIPT:?}" \
  --parent "$PARENT_LABEL=$PARENT_RECEIPT" --output "$PROVENANCE"
args=(--stage "$MODE" --arm "$ARM" --rotation "$ROTATION" --seed "$SEED" \
  --input-cache "$CACHE/input_cache.pt" --feature-cache "$CACHE/feature_cache.pt" \
  --target-cache "$CACHE/target_cache.pt" --provenance "$PROVENANCE" --output "$OUT")
if [[ "$MODE" == pilot ]]; then
  args+=(--latency-gate "$JEPA4D_LATENCY_GATE")
else
  args+=(--pilot-gate "$JEPA4D_PILOT_GATE")
fi
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/scripts/run_phase2f_training.py" "${args[@]}"
