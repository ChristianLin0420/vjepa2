#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'
umask 027

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${JEPA4D_PYTHON:-$ROOT/.conda-gpu/bin/python}"
[[ -x "$PYTHON" ]] || { printf 'missing executable Phase 2f Python: %s\n' "$PYTHON" >&2; exit 2; }
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || { printf 'Phase 2f submission requires a clean committed worktree\n' >&2; exit 2; }

ACCOUNT="edgeai_tao-ptm_image-foundation-model-clip"
PARTITIONS="polar4,polar3,polar,batch_block1,grizzly,batch_block2,batch_block3"
PREREGISTRATION="$ROOT/docs/experiments/2026-06-29-phase2f-scale-camera-preregistered.md"
SUN_ROOT="${JEPA4D_SUN_ROOT:-$ROOT/checkpoints/datasets/SUNRGBD}"
SUN_MANIFEST="${JEPA4D_SUN_MANIFEST:-$ROOT/jepa4d/config/benchmarks/manifests/sun_rgbd_phase2e_sensor_blocked_v1.yaml}"
VJEPA_CHECKPOINT="${JEPA4D_VJEPA_CHECKPOINT:-$ROOT/checkpoints/phase2b_assets/vjepa2.1-vitb-fpc64-384}"
VJEPA_IMPLEMENTATION="${JEPA4D_VJEPA_IMPLEMENTATION:-$ROOT/checkpoints/phase2b_assets/vjepa21_hf_impl}"
DIODE_ARCHIVE="${JEPA4D_DIODE_ARCHIVE:-$ROOT/checkpoints/datasets/DIODE/val.tar.gz}"
DIODE_DEVKIT_ROOT="${JEPA4D_DIODE_DEVKIT_ROOT:-$ROOT/checkpoints/datasets/DIODE/devkit}"
for required in "$PREREGISTRATION" "$SUN_MANIFEST" "$DIODE_ARCHIVE"; do
  [[ -f "$required" ]] || { printf 'missing required Phase 2f file: %s\n' "$required" >&2; exit 2; }
done
for required in "$SUN_ROOT" "$VJEPA_CHECKPOINT" "$VJEPA_IMPLEMENTATION" "$DIODE_DEVKIT_ROOT"; do
  [[ -d "$required" ]] || { printf 'missing required Phase 2f directory: %s\n' "$required" >&2; exit 2; }
done

COMMIT="$(git rev-parse HEAD)"
SHORT="$(git rev-parse --short=8 HEAD)"
EXECUTION_ID="${JEPA4D_EXECUTION_ID:-${SHORT}-$(date -u +%Y%m%dT%H%M%SZ)}"
GATE_ROOT="$ROOT/outputs/phase2f-gates/$EXECUTION_ID"
OUTPUT_ROOT="$ROOT/outputs/jepa4d_phase2f/$EXECUTION_ID"
GRAPH="$GATE_ROOT/dependency-graph.json"
TEST_RECEIPT="$GATE_ROOT/tests.json"
[[ ! -e "$GATE_ROOT" && ! -e "$OUTPUT_ROOT" ]] || { printf 'execution ID already exists: %s\n' "$EXECUTION_ID" >&2; exit 2; }
mkdir -p "$GATE_ROOT" "$OUTPUT_ROOT"

"$PYTHON" "$ROOT/slurm/phase2f_final_guard.py" registry-clear \
  --registry-root "$ROOT/outputs/jepa4d_phase2f" --preregistration "$PREREGISTRATION"

job_ids=()
graph_jobs=()
released=false
cleanup() {
  status=$?
  if [[ "$released" != true && ${#job_ids[@]} -gt 0 ]]; then
    joined="$(IFS=,; printf '%s' "${job_ids[*]}")"
    scancel "$joined" >/dev/null 2>&1 || true
  fi
  exit "$status"
}
trap cleanup EXIT

submit() {
  local name="$1" dependency="$2" exports="$3" sbatch_file="$4" result
  shift 4
  args=(--parsable --hold --job-name "$name" --export "$exports")
  [[ "$dependency" == - ]] || args+=(--dependency "afterok:$dependency")
  result="$(sbatch "${args[@]}" "$@" "$sbatch_file")"
  result="${result%%;*}"
  [[ "$result" =~ ^[0-9]+$ ]] || { printf 'invalid sbatch result: %s\n' "$result" >&2; return 2; }
  job_ids+=("$result")
  printf '%s\n' "$result"
}

register() {
  local label="$1" id="$2" name="$3" parents="$4" sbatch_file="$5" receipt="$6"
  graph_jobs+=("$label|$id|$name|$parents|$sbatch_file|$receipt")
}

common="ALL,JEPA4D_REPO_ROOT=$ROOT,JEPA4D_EXECUTION_ID=$EXECUTION_ID,JEPA4D_GRAPH=$GRAPH,JEPA4D_TEST_RECEIPT=$TEST_RECEIPT,JEPA4D_OUTPUT_ROOT=$OUTPUT_ROOT,JEPA4D_WANDB_ENTITY=crlc112358,JEPA4D_WANDB_PROJECT=jepa4d-worldmodel,WANDB_MODE=online,PYTHONUTF8=1"
assets="$common,JEPA4D_SUN_ROOT=$SUN_ROOT,JEPA4D_SUN_MANIFEST=$SUN_MANIFEST,JEPA4D_VJEPA_CHECKPOINT=$VJEPA_CHECKPOINT,JEPA4D_VJEPA_IMPLEMENTATION=$VJEPA_IMPLEMENTATION,JEPA4D_DIODE_ARCHIVE=$DIODE_ARCHIVE,JEPA4D_DIODE_DEVKIT_ROOT=$DIODE_DEVKIT_ROOT"

name="j4d-p2f-T-$SHORT"; T="$(submit "$name" - "$common" slurm/phase2f_tests.sbatch)"
job_ids+=("$T")
register T "$T" "$name" - slurm/phase2f_tests.sbatch "$TEST_RECEIPT"

name="j4d-p2f-A-$SHORT"; A_OUT="$OUTPUT_ROOT/assets"; A="$(submit "$name" "$T" "$assets,JEPA4D_STAGE_OUTPUT=$A_OUT" slurm/phase2f_asset_audit.sbatch)"
job_ids+=("$A")
register A "$A" "$name" T slurm/phase2f_asset_audit.sbatch "$A_OUT/asset_receipt.json"

name="j4d-p2f-C-$SHORT"; C_OUT="$OUTPUT_ROOT/cache"; C="$(submit "$name" "$T" "$assets,JEPA4D_STAGE_OUTPUT=$C_OUT" slurm/phase2f_cache.sbatch)"
job_ids+=("$C")
register C "$C" "$name" T slurm/phase2f_cache.sbatch "$C_OUT/cache_receipt.json"

name="j4d-p2f-Q-$SHORT"; Q_OUT="$OUTPUT_ROOT/static"; Q="$(submit "$name" "$C" "$assets,JEPA4D_STAGE_OUTPUT=$Q_OUT,JEPA4D_CACHE_ROOT=$C_OUT" slurm/phase2f_static_audit.sbatch)"
job_ids+=("$Q")
register Q "$Q" "$name" C slurm/phase2f_static_audit.sbatch "$Q_OUT/static_receipt.json"

latency_ids=()
for replicate in $(seq 0 11); do
  label="$(printf 'L%02d' "$replicate")"; name="j4d-p2f-${label,,}-$SHORT"
  rep_out="$OUTPUT_ROOT/latency/replicate-$(printf '%02d' "$replicate")"
  id="$(submit "$name" "$Q" "$assets,JEPA4D_REPLICATE=$replicate,JEPA4D_STAGE_OUTPUT=$rep_out,JEPA4D_CACHE_ROOT=$C_OUT,JEPA4D_STATIC_ROOT=$Q_OUT" slurm/phase2f_latency.sbatch)"
  job_ids+=("$id")
  latency_ids+=("$id")
  register "$label" "$id" "$name" Q slurm/phase2f_latency.sbatch "$rep_out/latency_receipt.json"
done
latency_dependency="$(IFS=:; printf '%s' "${latency_ids[*]}")"
latency_parents="$(for replicate in $(seq 0 11); do printf 'L%02d,' "$replicate"; done)"; latency_parents="${latency_parents%,}"
name="j4d-p2f-LA-$SHORT"; LA_OUT="$OUTPUT_ROOT/latency-aggregate"
LA="$(submit "$name" "$latency_dependency" "$common,JEPA4D_STAGE_OUTPUT=$LA_OUT,JEPA4D_LATENCY_ROOT=$OUTPUT_ROOT/latency" slurm/phase2f_latency_aggregate.sbatch)"
job_ids+=("$LA")
register LA "$LA" "$name" "$latency_parents" slurm/phase2f_latency_aggregate.sbatch "$LA_OUT/qualification.json"

pilot_ids=()
for index in 0 1 2 3; do
  arm="M$index"; label="P$index"; name="j4d-p2f-${label,,}-$SHORT"; pilot_out="$OUTPUT_ROOT/pilot/$arm"
  id="$(submit "$name" "$LA" "$common,JEPA4D_TRAIN_MODE=pilot,JEPA4D_ARM=$arm,JEPA4D_ROTATION=R0,JEPA4D_SEED=0,JEPA4D_JOB_LABEL=$label,JEPA4D_STAGE_OUTPUT=$pilot_out,JEPA4D_CACHE_ROOT=$C_OUT,JEPA4D_LATENCY_GATE=$LA_OUT/qualification.json" slurm/phase2f_train.sbatch)"
  job_ids+=("$id")
  pilot_ids+=("$id")
  register "$label" "$id" "$name" LA slurm/phase2f_train.sbatch "$pilot_out/training_receipt.json"
done
pilot_dependency="$(IFS=:; printf '%s' "${pilot_ids[*]}")"
name="j4d-p2f-PG-$SHORT"; PG_OUT="$OUTPUT_ROOT/pilot-gate"
PG="$(submit "$name" "$pilot_dependency" "$common,JEPA4D_STAGE_OUTPUT=$PG_OUT,JEPA4D_PILOT_ROOT=$OUTPUT_ROOT/pilot,JEPA4D_LATENCY_GATE=$LA_OUT/qualification.json" slurm/phase2f_pilot_gate.sbatch)"
job_ids+=("$PG")
register PG "$PG" "$name" P0,P1,P2,P3 slurm/phase2f_pilot_gate.sbatch "$PG_OUT/qualification.json"

formal_ids=()
formal_labels=()
for arm in M0 M1 M2 M3; do
  for rotation in R0 R1 R2 R3; do
    for seed in 0 1 2; do
      label="F-$arm-$rotation-S$seed"; formal_labels+=("$label")
      name="j4d-p2f-f-${arm,,}-${rotation,,}-s$seed-$SHORT"
      formal_out="$OUTPUT_ROOT/formal/$arm/$rotation/seed-$seed"
      id="$(submit "$name" "$PG" "$common,JEPA4D_TRAIN_MODE=formal,JEPA4D_ARM=$arm,JEPA4D_ROTATION=$rotation,JEPA4D_SEED=$seed,JEPA4D_JOB_LABEL=$label,JEPA4D_STAGE_OUTPUT=$formal_out,JEPA4D_CACHE_ROOT=$C_OUT,JEPA4D_PILOT_GATE=$PG_OUT/qualification.json" slurm/phase2f_train.sbatch)"
      job_ids+=("$id")
      formal_ids+=("$id")
      register "$label" "$id" "$name" PG slurm/phase2f_train.sbatch "$formal_out/training_receipt.json"
    done
  done
done
formal_dependency="$(IFS=:; printf '%s' "${formal_ids[*]}")"
formal_parents="$(IFS=,; printf '%s' "${formal_labels[*]}")"
name="j4d-p2f-S-$SHORT"; S_OUT="$OUTPUT_ROOT/selection"
S="$(submit "$name" "$formal_dependency" "$common,JEPA4D_STAGE_OUTPUT=$S_OUT,JEPA4D_FORMAL_ROOT=$OUTPUT_ROOT/formal,JEPA4D_LATENCY_GATE=$LA_OUT/qualification.json,JEPA4D_PILOT_GATE=$PG_OUT/qualification.json" slurm/phase2f_select.sbatch)"
job_ids+=("$S")
register S "$S" "$name" "$formal_parents" slurm/phase2f_select.sbatch "$S_OUT/selector.json"

name="j4d-p2f-E-$SHORT"; E_OUT="$OUTPUT_ROOT/final"
E="$(submit "$name" "$S:$A" "$assets,JEPA4D_STAGE_OUTPUT=$E_OUT,JEPA4D_SELECTOR_RECEIPT=$S_OUT/selector.json,JEPA4D_ASSET_RECEIPT=$A_OUT/asset_receipt.json" slurm/phase2f_final.sbatch)"
job_ids+=("$E")
register E "$E" "$name" S,A slurm/phase2f_final.sbatch "$E_OUT/final_receipt.json"

name="j4d-p2f-Z-$SHORT"; Z_OUT="$OUTPUT_ROOT/postflight"
Z="$(submit "$name" "$E" "$common,JEPA4D_STAGE_OUTPUT=$Z_OUT,JEPA4D_FINAL_RECEIPT=$E_OUT/final_receipt.json" slurm/phase2f_postflight.sbatch)"
job_ids+=("$Z")
register Z "$Z" "$name" E slurm/phase2f_postflight.sbatch "$Z_OUT/postflight_receipt.json"

graph_args=()
for specification in "${graph_jobs[@]}"; do graph_args+=(--job "$specification"); done
"$PYTHON" scripts/write_phase2f_dependency_graph.py \
  --repo-root "$ROOT" --execution-id "$EXECUTION_ID" --preregistration "$PREREGISTRATION" \
  --test-receipt "$TEST_RECEIPT" --output-root "$OUTPUT_ROOT" --output "$GRAPH" "${graph_args[@]}"
[[ -s "$GRAPH" ]] || { printf 'canonical Phase 2f graph was not written\n' >&2; exit 2; }
for ((offset=0; offset<${#job_ids[@]}; offset+=20)); do
  chunk=("${job_ids[@]:offset:20}")
  joined="$(IFS=,; printf '%s' "${chunk[*]}")"
  scontrol release "$joined"
done
released=true
trap - EXIT
printf 'Phase 2f execution: %s\nGraph: %s\nTest job: %s\nFinal job: %s\nPostflight job: %s\nJobs: %s\nCommit: %s\nAccount: %s\nPartitions: %s\n' \
  "$EXECUTION_ID" "$GRAPH" "$T" "$E" "$Z" "${#job_ids[@]}" "$COMMIT" "$ACCOUNT" "$PARTITIONS"
