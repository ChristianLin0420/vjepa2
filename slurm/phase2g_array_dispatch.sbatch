#!/usr/bin/env bash
#SBATCH --job-name=p2gq-array-submit-required
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=polar4,polar3,polar,batch_block1,grizzly,batch_block2,batch_block3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --signal=B:USR1@120

set -Eeuo pipefail
IFS=$'\n\t'
umask 027
ROOT="${JEPA4D_REPO_ROOT:?}"
STAGE="${JEPA4D_DISPATCH_STAGE:?}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?}"
ARRAY_ID="${SLURM_ARRAY_JOB_ID:?}"
SHORT="${JEPA4D_SHORT_COMMIT:?}"
OUTPUT_ROOT="${JEPA4D_OUTPUT_ROOT:?}"
CACHE_ROOT="${JEPA4D_CACHE_ROOT:?}"
[[ "$TASK_ID" =~ ^[0-9]+$ && "$ARRAY_ID" =~ ^[0-9]+$ ]] || { printf 'invalid array identity\n' >&2; exit 2; }
(( TASK_ID >= 0 && TASK_ID < 48 )) || { printf 'array task out of range: %s\n' "$TASK_ID" >&2; exit 2; }
export JEPA4D_LOGICAL_JOB_ID="${ARRAY_ID}_${TASK_ID}"
arm_index=$((TASK_ID / 12))
within_arm=$((TASK_ID % 12))
rotation_index=$((within_arm / 3))
slot=$((within_arm % 3))
export JEPA4D_ARM="M$arm_index"
export JEPA4D_ROTATION="R$rotation_index"

case "$JEPA4D_ROTATION" in
  R0) heldout=kv2 ;;
  R1) heldout=kv1 ;;
  R2) heldout=xtion ;;
  R3) heldout=realsense ;;
  *) printf 'unknown rotation\n' >&2; exit 2 ;;
esac

case "$STAGE" in
  tuning)
    learning_rates=(0.0005 0.001 0.002)
    export JEPA4D_LEARNING_RATE="${learning_rates[$slot]}"
    export JEPA4D_LR_INDEX="$slot"
    export JEPA4D_JOB_LABEL="H-$JEPA4D_ARM-$JEPA4D_ROTATION-L$slot"
    export JEPA4D_STAGE_OUTPUT="$OUTPUT_ROOT/tuning/$JEPA4D_ARM/$JEPA4D_ROTATION/lr-$slot"
    export JEPA4D_CACHE_VIEW="$CACHE_ROOT/rotations/$JEPA4D_ROTATION"
    export SLURM_JOB_NAME="p2gq-H-$JEPA4D_ARM-$JEPA4D_ROTATION-L${slot}-$SHORT"
    unset JEPA4D_CACHE_ROOT CACHE_ROOT
    exec bash "$ROOT/slurm/phase2g_tune.sbatch"
    ;;
  formal)
    export JEPA4D_SEED="$slot"
    export JEPA4D_JOB_LABEL="F-$JEPA4D_ARM-$JEPA4D_ROTATION-S$slot"
    export JEPA4D_STAGE_OUTPUT="$OUTPUT_ROOT/formal/$JEPA4D_ARM/$JEPA4D_ROTATION/seed-$slot"
    export JEPA4D_CACHE_VIEW="$CACHE_ROOT/rotations/$JEPA4D_ROTATION"
    export SLURM_JOB_NAME="p2gq-F-$JEPA4D_ARM-$JEPA4D_ROTATION-S${slot}-$SHORT"
    unset JEPA4D_CACHE_ROOT CACHE_ROOT
    exec bash "$ROOT/slurm/phase2g_train.sbatch"
    ;;
  evaluation)
    export JEPA4D_SEED="$slot"
    export JEPA4D_HELDOUT_FAMILY="$heldout"
    export JEPA4D_JOB_LABEL="V-$JEPA4D_ARM-$JEPA4D_ROTATION-S$slot"
    export JEPA4D_STAGE_OUTPUT="$OUTPUT_ROOT/evaluation/$JEPA4D_ARM/$JEPA4D_ROTATION/seed-$slot"
    export JEPA4D_INPUT_SHARD="$CACHE_ROOT/shards/$heldout/input.pt"
    export JEPA4D_FEATURE_SHARD="$CACHE_ROOT/shards/$heldout/feature.pt"
    export JEPA4D_TARGET_SHARD="$CACHE_ROOT/shards/$heldout/target.pt"
    export JEPA4D_TRAINING_RECEIPT="$OUTPUT_ROOT/formal/$JEPA4D_ARM/$JEPA4D_ROTATION/seed-$slot/training_receipt.json"
    export JEPA4D_PARENT_LABEL="F-$JEPA4D_ARM-$JEPA4D_ROTATION-S$slot"
    export SLURM_JOB_NAME="p2gq-V-$JEPA4D_ARM-$JEPA4D_ROTATION-S${slot}-$SHORT"
    unset JEPA4D_CACHE_ROOT CACHE_ROOT
    exec bash "$ROOT/slurm/phase2g_evaluate.sbatch"
    ;;
  *) printf 'unknown Phase 2g array stage: %s\n' "$STAGE" >&2; exit 2 ;;
esac
