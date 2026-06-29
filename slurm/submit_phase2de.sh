#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ -x "$ROOT/.conda-gpu/bin/python" ]] || { printf 'missing .conda-gpu\n' >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { printf 'submission requires a clean Git worktree\n' >&2; exit 2; }

PHASE2C_OUTPUT="${JEPA4D_PHASE2C_OUTPUT:-$ROOT/outputs/jepa4d_phase2c/tum_rgbd_cross_sequence_9a8f8f0cb5fb-20260629T120903Z}"
DATASET_PARENT="${JEPA4D_DATASET_PARENT:-$ROOT/checkpoints/datasets}"
SUNRGBD_ROOT="${JEPA4D_SUNRGBD_ROOT:-$ROOT/checkpoints/datasets/SUNRGBD}"
TEST_RECEIPT="${JEPA4D_TEST_REPORT:-$ROOT/outputs/phase2d-gates/tests.json}"
P2D_GRAPH="${JEPA4D_PHASE2D_GRAPH:-$ROOT/outputs/phase2d-gates/dependency-graph.json}"
P2E_GRAPH="${JEPA4D_PHASE2E_GRAPH:-$ROOT/outputs/phase2e-gates/dependency-graph.json}"
SHORT="$(git rev-parse --short=8 HEAD)"

submit() {
  local result
  result="$(sbatch --parsable "$@")"
  printf '%s\n' "${result%%;*}"
}

common="ALL,JEPA4D_REPO_ROOT=$ROOT,JEPA4D_TEST_REPORT=$TEST_RECEIPT,JEPA4D_WANDB_PROJECT=jepa4d-worldmodel"
p2c="$common,JEPA4D_PHASE2C_OUTPUT=$PHASE2C_OUTPUT,JEPA4D_DATASET_PARENT=$DATASET_PARENT"

test_job="$(submit --job-name="j4d-p2de-test-$SHORT" --export="$common" slurm/phase2d_tests.sbatch)"
attribution_job="$(submit --job-name="j4d-p2d-attr-$SHORT" --dependency="afterok:$test_job" \
  --export="$p2c,JEPA4D_OUTPUT=$ROOT/outputs/jepa4d_phase2d/fusion-attribution" slurm/phase2d_attribution.sbatch)"
calibration_job="$(submit --job-name="j4d-p2d-cal-$SHORT" --dependency="afterok:$attribution_job" \
  --export="$p2c,JEPA4D_ATTRIBUTION_OUTPUT=$ROOT/outputs/jepa4d_phase2d/fusion-attribution,JEPA4D_OUTPUT=$ROOT/outputs/jepa4d_phase2d/calibration-scale-audit" \
  slurm/phase2d_calibration_audit.sbatch)"

latency_jobs=()
for replicate in $(seq 0 11); do
  label="$(printf '%02d' "$replicate")"
  latency_jobs+=("$(submit --job-name="j4d-p2d-lat${label}-$SHORT" --dependency="afterok:$test_job" \
    --export="$p2c,JEPA4D_REPLICATE=$replicate,JEPA4D_OUTPUT_ROOT=$ROOT/outputs/jepa4d_phase2d/latency" \
    slurm/phase2d_latency.sbatch)")
done
latency_dependency="$(IFS=:; printf '%s' "${latency_jobs[*]}")"
latency_csv="$(IFS=,; printf '%s' "${latency_jobs[*]}")"
latency_aggregate_job="$(submit --job-name="j4d-p2d-latagg-$SHORT" \
  --dependency="afterok:$latency_dependency" \
  --export="$common,JEPA4D_INPUT_ROOT=$ROOT/outputs/jepa4d_phase2d/latency,JEPA4D_OUTPUT=$ROOT/outputs/jepa4d_phase2d/latency-aggregate" \
  slurm/phase2d_latency_aggregate.sbatch)"

cache_job="$(submit --job-name="j4d-p2e-cache-$SHORT" --dependency="afterok:$test_job" \
  --export="$common,JEPA4D_SUNRGBD_ROOT=$SUNRGBD_ROOT,JEPA4D_OUTPUT=$ROOT/outputs/jepa4d_phase2e/sunrgbd-cache" \
  slurm/phase2e_cache.sbatch)"
pilot_job="$(submit --job-name="j4d-p2e-pilot-$SHORT" --dependency="afterok:$cache_job" \
  --export="$common,JEPA4D_TRAIN_MODE=pilot,JEPA4D_SHARD_ID=pilot,JEPA4D_OUTPUT_ROOT=$ROOT/outputs/jepa4d_phase2e/pilot" \
  slurm/phase2e_train.sbatch)"

formal_jobs=()
for shard in 0 1 2 3; do
  formal_jobs+=("$(submit --job-name="j4d-p2e-s${shard}-$SHORT" --dependency="afterok:$pilot_job" \
    --export="$common,JEPA4D_TRAIN_MODE=formal,JEPA4D_SHARD_ID=$shard,JEPA4D_OUTPUT_ROOT=$ROOT/outputs/jepa4d_phase2e/formal,JEPA4D_PILOT_OUTPUT=$ROOT/outputs/jepa4d_phase2e/pilot/shard-pilot" \
    slurm/phase2e_train.sbatch)")
done
formal_dependency="$(IFS=:; printf '%s' "${formal_jobs[*]}")"
formal_csv="$(IFS=,; printf '%s' "${formal_jobs[*]}")"

phase2d_aggregate_job="$(submit --job-name="j4d-p2d-report-$SHORT" \
  --dependency="afterok:$calibration_job:$latency_aggregate_job" \
  --export="$common,JEPA4D_DEPENDENCY_GRAPH=$P2D_GRAPH,JEPA4D_ATTRIBUTION_OUTPUT=$ROOT/outputs/jepa4d_phase2d/fusion-attribution,JEPA4D_CALIBRATION_OUTPUT=$ROOT/outputs/jepa4d_phase2d/calibration-scale-audit,JEPA4D_LATENCY_OUTPUT=$ROOT/outputs/jepa4d_phase2d/latency-aggregate,JEPA4D_LATENCY_ROOT=$ROOT/outputs/jepa4d_phase2d/latency,JEPA4D_OUTPUT=$ROOT/outputs/jepa4d_phase2d/aggregate" \
  slurm/phase2d_aggregate.sbatch)"
final_job="$(submit --job-name="j4d-p2e-final-$SHORT" --dependency="afterok:$formal_dependency" \
  --export="$common,JEPA4D_DEPENDENCY_GRAPH=$P2E_GRAPH,JEPA4D_CACHE_ROOT=$ROOT/outputs/jepa4d_phase2e/sunrgbd-cache,JEPA4D_FORMAL_ROOT=$ROOT/outputs/jepa4d_phase2e/formal,JEPA4D_OUTPUT=$ROOT/outputs/jepa4d_phase2e/final" \
  slurm/phase2e_final.sbatch)"

"$ROOT/.conda-gpu/bin/python" scripts/write_phase2_dependency_graphs.py \
  --phase2d-output "$P2D_GRAPH" --phase2e-output "$P2E_GRAPH" \
  --test-job "$test_job" --attribution-job "$attribution_job" --calibration-job "$calibration_job" \
  --latency-jobs "$latency_csv" --latency-aggregate-job "$latency_aggregate_job" \
  --phase2d-aggregate-job "$phase2d_aggregate_job" --cache-job "$cache_job" --pilot-job "$pilot_job" \
  --formal-jobs "$formal_csv" --final-job "$final_job"

printf 'Phase2d report job: %s\nPhase2e final job: %s\n' "$phase2d_aggregate_job" "$final_job"
