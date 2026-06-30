#!/usr/bin/env bash
#SBATCH --job-name=p2f8-array
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

[[ "$TASK_ID" =~ ^[0-9]+$ && "$ARRAY_ID" =~ ^[0-9]+$ ]] || {
  printf 'invalid Slurm array identity: %s_%s\n' "$ARRAY_ID" "$TASK_ID" >&2
  exit 2
}
export JEPA4D_LOGICAL_JOB_ID="${ARRAY_ID}_${TASK_ID}"

case "$STAGE" in
  latency)
    (( TASK_ID >= 0 && TASK_ID < 12 )) || { printf 'latency task out of range: %s\n' "$TASK_ID" >&2; exit 2; }
    export JEPA4D_REPLICATE="$TASK_ID"
    export JEPA4D_STAGE_OUTPUT
    JEPA4D_STAGE_OUTPUT="$(printf '%s/latency/replicate-%02d' "$OUTPUT_ROOT" "$TASK_ID")"
    export SLURM_JOB_NAME="p2f8-L$(printf '%02d' "$TASK_ID")-$SHORT"
    exec bash "$ROOT/slurm/phase2f_latency.sbatch"
    ;;
  pilot)
    (( TASK_ID >= 0 && TASK_ID < 4 )) || { printf 'pilot task out of range: %s\n' "$TASK_ID" >&2; exit 2; }
    export JEPA4D_TRAIN_MODE=pilot
    export JEPA4D_ARM="M$TASK_ID"
    export JEPA4D_ROTATION=R0
    export JEPA4D_SEED=0
    export JEPA4D_JOB_LABEL="P$TASK_ID"
    export JEPA4D_STAGE_OUTPUT="$OUTPUT_ROOT/pilot/M$TASK_ID"
    export SLURM_JOB_NAME="p2f8-P${TASK_ID}-$SHORT"
    exec bash "$ROOT/slurm/phase2f_train.sbatch"
    ;;
  formal)
    (( TASK_ID >= 0 && TASK_ID < 48 )) || { printf 'formal task out of range: %s\n' "$TASK_ID" >&2; exit 2; }
    arm_index=$((TASK_ID / 12))
    within_arm=$((TASK_ID % 12))
    rotation_index=$((within_arm / 3))
    seed=$((within_arm % 3))
    export JEPA4D_TRAIN_MODE=formal
    export JEPA4D_ARM="M$arm_index"
    export JEPA4D_ROTATION="R$rotation_index"
    export JEPA4D_SEED="$seed"
    export JEPA4D_JOB_LABEL="F-$JEPA4D_ARM-$JEPA4D_ROTATION-S$seed"
    export JEPA4D_STAGE_OUTPUT="$OUTPUT_ROOT/formal/$JEPA4D_ARM/$JEPA4D_ROTATION/seed-$seed"
    export SLURM_JOB_NAME="p2f8-F-$JEPA4D_ARM-$JEPA4D_ROTATION-S${seed}-$SHORT"
    exec bash "$ROOT/slurm/phase2f_train.sbatch"
    ;;
  *)
    printf 'unknown Phase 2f array dispatch stage: %s\n' "$STAGE" >&2
    exit 2
    ;;
esac
