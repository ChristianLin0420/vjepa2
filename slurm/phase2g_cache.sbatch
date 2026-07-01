#!/usr/bin/env bash
#SBATCH --job-name=p2gq-C-submit-required
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
jepa4d_init_log_dir "$REPO_ROOT" "phase2g-cache"
jepa4d_install_traps
jepa4d_capture_environment "$REPO_ROOT" "$JEPA4D_JOB_LOG_DIR"
jepa4d_start_gpu_monitor "$JEPA4D_JOB_LOG_DIR/gpu-telemetry.csv"
export JEPA4D_GPU_TELEMETRY_PATH="$JEPA4D_JOB_LOG_DIR/gpu-telemetry.csv"
export WANDB_MODE=online PYTHONUTF8=1
OUT="${JEPA4D_STAGE_OUTPUT:?}"
MATERIALIZED="${JEPA4D_SUN_MATERIALIZATION:?}"
PROVENANCE="$JEPA4D_JOB_LOG_DIR/execution_provenance.json"
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/slurm/phase2g_contract.py" emit-provenance \
  --graph "${JEPA4D_GRAPH:?}" --label C --test-receipt "${JEPA4D_TEST_RECEIPT:?}" \
  --parent "T=${JEPA4D_TEST_RECEIPT}" --source "preregistration=${JEPA4D_PREREGISTRATION:?}" \
  --output "$PROVENANCE"
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/scripts/materialize_phase2g_sun.py" \
  --archive "${JEPA4D_SUN_ARCHIVE:?}" --provenance "$PROVENANCE" --output "$MATERIALIZED"
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/scripts/build_phase2g_data_cache.py" \
  --sun-root "$MATERIALIZED/SUNRGBD" --materialization-receipt "$MATERIALIZED/materialization_receipt.json" \
  --vjepa-checkpoint "${JEPA4D_VJEPA_CHECKPOINT:?}" \
  --vjepa-implementation "${JEPA4D_VJEPA_IMPLEMENTATION:?}" --provenance "$PROVENANCE" --output "$OUT"
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/slurm/phase2g_contract.py" verify-receipt \
  --receipt "$OUT/cache_receipt.json" --provenance "$PROVENANCE"
