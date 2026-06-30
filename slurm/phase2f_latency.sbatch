#!/usr/bin/env bash
#SBATCH --job-name=j4d-p2f-latency
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=polar4,polar3,polar,batch_block1,grizzly,batch_block2,batch_block3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
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
jepa4d_init_log_dir "$REPO_ROOT" "phase2f-latency"
jepa4d_install_traps
jepa4d_capture_environment "$REPO_ROOT" "$JEPA4D_JOB_LOG_DIR"
jepa4d_start_gpu_monitor "$JEPA4D_JOB_LOG_DIR/gpu-telemetry.csv"
export WANDB_MODE=online PYTHONUTF8=1
REPLICA="${JEPA4D_REPLICATE:?}"
LABEL="$(printf 'L%02d' "$REPLICA")"
OUT="${JEPA4D_STAGE_OUTPUT:?}"
CACHE="${JEPA4D_CACHE_ROOT:?}"
STATIC="${JEPA4D_STATIC_ROOT:?}"
PROVENANCE="$JEPA4D_JOB_LOG_DIR/execution_provenance.json"
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/slurm/phase2f_contract.py" emit-provenance \
  --graph "${JEPA4D_GRAPH:?}" --label "$LABEL" --test-receipt "${JEPA4D_TEST_RECEIPT:?}" \
  --parent "Q=$STATIC/static_receipt.json" --output "$PROVENANCE"
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/scripts/run_phase2f_latency.py" \
  --replica "$REPLICA" --input-cache "$CACHE/input_cache.pt" --feature-cache "$CACHE/feature_cache.pt" \
  --vjepa-checkpoint "${JEPA4D_VJEPA_CHECKPOINT:?}" --vjepa-implementation "${JEPA4D_VJEPA_IMPLEMENTATION:?}" \
  --provenance "$PROVENANCE" --output "$OUT"
