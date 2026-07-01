#!/usr/bin/env bash
#SBATCH --job-name=p2gq-O-submit-required
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=polar4,polar3,polar,batch_block1,grizzly,batch_block2,batch_block3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
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
jepa4d_init_log_dir "$REPO_ROOT" "phase2g-opacity"
jepa4d_install_traps
jepa4d_capture_environment "$REPO_ROOT" "$JEPA4D_JOB_LOG_DIR"
export WANDB_MODE=online PYTHONUTF8=1
OUT="${JEPA4D_STAGE_OUTPUT:?}"
PROVENANCE="$JEPA4D_JOB_LOG_DIR/execution_provenance.json"
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/slurm/phase2g_contract.py" emit-provenance \
  --graph "${JEPA4D_GRAPH:?}" --label O --test-receipt "${JEPA4D_TEST_RECEIPT:?}" \
  --parent "T=${JEPA4D_TEST_RECEIPT}" --output "$PROVENANCE"
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/slurm/phase2g_stage_gate.py" opacity \
  --provenance "$PROVENANCE" --output "$OUT"
