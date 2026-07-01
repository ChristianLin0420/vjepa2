#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'
umask 027

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }

DRY_RUN=false
case "${1:-}" in
  "") ;;
  --dry-run) DRY_RUN=true ;;
  *) die "usage: $0 [--dry-run]" ;;
esac
(( $# <= 1 )) || die "usage: $0 [--dry-run]"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
PYTHON="${JEPA4D_PYTHON:-$ROOT/.conda-gpu/bin/python}"
[[ -x "$PYTHON" ]] || die "missing executable Phase 2g Python: $PYTHON"
PYTHON="$(readlink -f -- "$PYTHON")"
[[ -x "$PYTHON" ]] || die "unable to canonicalize Phase 2g Python"
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "submission requires a Git worktree"
COMMIT="$(git rev-parse --verify 'HEAD^{commit}')" || die "submission requires a committed HEAD"
[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || die "formal Phase 2g requires a clean committed worktree"
BRANCH="$(git branch --show-current)"
[[ -n "$BRANCH" ]] || die "formal Phase 2g forbids detached HEAD"
UPSTREAM="$(git rev-parse '@{u}' 2>/dev/null)" || die "formal Phase 2g requires a configured pushed upstream"
UPSTREAM_NAME="$(git rev-parse --abbrev-ref '@{u}')" || die "formal Phase 2g requires a named pushed upstream"
[[ "$COMMIT" == "$UPSTREAM" ]] || die "formal Phase 2g HEAD must equal its pushed upstream"
SHORT="$(git rev-parse --short=8 "$COMMIT")"

REGISTRY="$ROOT/configs/validation/dataset_registry.yaml"
LEDGER="$ROOT/configs/validation/consumed_test_ledger.yaml"
READINESS="$ROOT/configs/validation/geometry/phase2_readiness_v1.yaml"
PREREGISTRATION="${JEPA4D_PHASE2G_PREREGISTRATION:-$ROOT/docs/experiments/2026-06-30-phase2g-quality-first-preregistered.md}"
SUN_ARCHIVE="${JEPA4D_SUN_ARCHIVE:?export JEPA4D_SUN_ARCHIVE before Phase 2g preflight}"
VJEPA_CHECKPOINT="${JEPA4D_VJEPA_CHECKPOINT:?export JEPA4D_VJEPA_CHECKPOINT before Phase 2g preflight}"
VJEPA_IMPLEMENTATION="${JEPA4D_VJEPA_IMPLEMENTATION:?export JEPA4D_VJEPA_IMPLEMENTATION before Phase 2g preflight}"
for file in "$REGISTRY" "$LEDGER" "$READINESS" "$PREREGISTRATION" "$SUN_ARCHIVE"; do
  [[ -f "$file" && ! -L "$file" ]] || die "missing regular Phase 2g input file: $file"
done
for directory in "$VJEPA_CHECKPOINT" "$VJEPA_IMPLEMENTATION"; do
  [[ -d "$directory" && ! -L "$directory" ]] || die "missing Phase 2g input directory: $directory"
done
SUN_ARCHIVE="$(readlink -f -- "$SUN_ARCHIVE")"
VJEPA_CHECKPOINT="$(readlink -f -- "$VJEPA_CHECKPOINT")"
VJEPA_IMPLEMENTATION="$(readlink -f -- "$VJEPA_IMPLEMENTATION")"

WANDB_ENTITY="${JEPA4D_WANDB_ENTITY:-crlc112358}"
WANDB_PROJECT="${JEPA4D_WANDB_PROJECT:-jepa4d-worldmodel}"
VALIDATION_STATE_ROOT="${JEPA4D_VALIDATION_STATE_ROOT:-$ROOT/outputs/validation-state}"
[[ "$VALIDATION_STATE_ROOT" == /* ]] || VALIDATION_STATE_ROOT="$ROOT/$VALIDATION_STATE_ROOT"
export JEPA4D_VALIDATION_STATE_ROOT="$VALIDATION_STATE_ROOT"
[[ "$WANDB_ENTITY" == "crlc112358" ]] || die "formal Phase 2g W&B entity is frozen to crlc112358"
[[ "$WANDB_PROJECT" == "jepa4d-worldmodel" ]] || die "formal Phase 2g W&B project is frozen to jepa4d-worldmodel"
JOB_HOME="${HOME:?HOME is required for W&B authentication}"
NETRC="$JOB_HOME/.netrc"
[[ -f "$NETRC" && ! -L "$NETRC" ]] || die "W&B authentication requires a regular HOME/.netrc"
[[ "$(stat -Lc '%a' "$NETRC")" == "600" ]] || die "HOME/.netrc must have mode 0600"
[[ "$(stat -Lc '%u' "$NETRC")" == "$(id -u)" ]] || die "HOME/.netrc must be owned by the submitting user"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
NONCE="$(printf '%04x%04x' "$RANDOM" "$RANDOM")"
EXECUTION_ID="${JEPA4D_EXECUTION_ID:-p2gq-${SHORT}-${STAMP}-${NONCE}}"
[[ "$EXECUTION_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$ ]] || die "invalid JEPA4D_EXECUTION_ID"
STATE_ROOT="${JEPA4D_PHASE2G_STATE_ROOT:-$ROOT/outputs}"
[[ "$STATE_ROOT" == /* ]] || STATE_ROOT="$ROOT/$STATE_ROOT"
GATE_ROOT="$STATE_ROOT/phase2g-gates/$EXECUTION_ID"
OUTPUT_ROOT="$STATE_ROOT/jepa4d_phase2g/$EXECUTION_ID"
LOG_ROOT="$STATE_ROOT/slurm_logs/$EXECUTION_ID"
GRAPH="$GATE_ROOT/dependency-graph.json"
TEST_RECEIPT="$GATE_ROOT/tests.json"
[[ ! -e "$GATE_ROOT" && ! -L "$GATE_ROOT" && ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" \
  && ! -e "$LOG_ROOT" && ! -L "$LOG_ROOT" ]] || \
  die "fresh execution paths already exist: $EXECUTION_ID"

if [[ "$DRY_RUN" != true && "${JEPA4D_EXECUTION_WORKTREE:-}" != 1 ]]; then
  command -v squeue >/dev/null 2>&1 || die "squeue is required"
  active="$(squeue -r -h -u "$(id -un)" -o '%i' | awk 'NF {print $1}' | sort -u)" || die "unable to query active jobs"
  active_count="$(printf '%s\n' "$active" | awk 'NF {count++} END {print count+0}')"
  (( active_count == 0 )) || die "formal Phase 2g requires zero pre-existing active tasks; found $active_count"
  preregistration_relative="$(realpath --relative-to="$ROOT" "$PREREGISTRATION")"
  [[ "$preregistration_relative" != ../* && "$preregistration_relative" != .. ]] || \
    die "formal preregistration must be tracked inside the source worktree"
  worktree_root="$STATE_ROOT/phase2g-worktrees/$EXECUTION_ID"
  execution_branch="phase2g-exec-${SHORT}-${NONCE}"
  anchor_branch="phase2g-pushed-${SHORT}-${NONCE}"
  [[ ! -e "$worktree_root" && ! -L "$worktree_root" ]] || die "execution worktree already exists: $worktree_root"
  mkdir -p "$(dirname "$worktree_root")"
  worktree_created=false
  anchor_created=false
  cleanup_worktree() {
    status=$?
    if [[ -e "$GATE_ROOT/release-attempted" ]]; then
      printf 'retaining execution worktree after an ambiguous scheduler-release outcome: %s\n' \
        "$worktree_root" >&2
      exit "$status"
    fi
    if [[ "$worktree_created" == true ]]; then
      git -C "$ROOT" worktree remove --force "$worktree_root" >/dev/null 2>&1 || true
      git -C "$ROOT" branch -D "$execution_branch" >/dev/null 2>&1 || true
    fi
    if [[ "$anchor_created" == true ]]; then
      git -C "$ROOT" branch -D "$anchor_branch" >/dev/null 2>&1 || true
    fi
    exit "$status"
  }
  trap cleanup_worktree EXIT
  git branch "$anchor_branch" "$COMMIT"
  anchor_created=true
  git worktree add -b "$execution_branch" "$worktree_root" "$COMMIT"
  worktree_created=true
  git -C "$worktree_root" branch --set-upstream-to="$anchor_branch" "$execution_branch"
  set +e
  JEPA4D_EXECUTION_WORKTREE=1 \
  JEPA4D_EXECUTION_ID="$EXECUTION_ID" \
  JEPA4D_PHASE2G_STATE_ROOT="$STATE_ROOT" \
  JEPA4D_PHASE2G_PREREGISTRATION="$worktree_root/$preregistration_relative" \
  JEPA4D_VALIDATION_STATE_ROOT="$VALIDATION_STATE_ROOT" \
  JEPA4D_SUN_ARCHIVE="$SUN_ARCHIVE" \
  JEPA4D_VJEPA_CHECKPOINT="$VJEPA_CHECKPOINT" \
  JEPA4D_VJEPA_IMPLEMENTATION="$VJEPA_IMPLEMENTATION" \
  JEPA4D_PYTHON="$PYTHON" \
    bash "$worktree_root/slurm/submit_phase2g.sh"
  child_status=$?
  set -e
  if (( child_status == 0 )); then
    worktree_created=false
    anchor_created=false
    trap - EXIT
    printf 'execution_worktree=%s\nexecution_branch=%s\npushed_anchor_branch=%s\n' \
      "$worktree_root" "$execution_branch" "$anchor_branch"
    exit 0
  fi
  exit "$child_status"
fi

TEMP_ROOT=""
if [[ "$DRY_RUN" == true ]]; then
  TEMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/phase2g-dry-run.XXXXXX")"
  PREFLIGHT="$TEMP_ROOT/preflight.json"
else
  PREFLIGHT="$GATE_ROOT/preflight.json"
fi
cleanup_temp() { [[ -z "$TEMP_ROOT" ]] || rm -rf "$TEMP_ROOT"; }
trap cleanup_temp EXIT

"$PYTHON" "$ROOT/slurm/phase2g_preflight.py" \
  --repo-root "$ROOT" --execution-id "$EXECUTION_ID" --registry "$REGISTRY" --ledger "$LEDGER" --readiness "$READINESS" \
  --preregistration "$PREREGISTRATION" --sun-archive "$SUN_ARCHIVE" \
  --vjepa-checkpoint "$VJEPA_CHECKPOINT" --vjepa-implementation "$VJEPA_IMPLEMENTATION" \
  --wandb-entity "$WANDB_ENTITY" --wandb-project "$WANDB_PROJECT" --output "$PREFLIGHT"

submission_ids=()
graph_jobs=()
released=false
cancel_held() {
  status=$?
  if [[ "$DRY_RUN" != true && "$released" != true && ${#submission_ids[@]} -gt 0 ]]; then
    joined="$(IFS=,; printf '%s' "${submission_ids[*]}")"
    scancel "$joined" >/dev/null 2>&1 || true
  fi
  cleanup_temp
  exit "$status"
}
trap cancel_held EXIT

submit() {
  local name="$1" dependency="$2" exports="$3" sbatch_file="$4" result
  shift 4
  if [[ "$DRY_RUN" == true ]]; then
    case "$name" in
      p2gq-T-*) printf '900001\n' ;;
      p2gq-O-*) printf '900002\n' ;;
      p2gq-C-*) printf '900003\n' ;;
      p2gq-Q-*) printf '900004\n' ;;
      p2gq-HG-*) printf '900006\n' ;;
      p2gq-H-*) printf '900005\n' ;;
      p2gq-F-*) printf '900007\n' ;;
      p2gq-V-*) printf '900008\n' ;;
      p2gq-S-*) printf '900009\n' ;;
      p2gq-G-*) printf '900010\n' ;;
      p2gq-Z-*) printf '900011\n' ;;
      *) die "unknown dry-run submission name: $name" ;;
    esac
    return
  fi
  local -a args=(--parsable --hold --job-name "$name" --export "$exports")
  [[ "$dependency" == - ]] || args+=(--dependency "afterok:$dependency")
  result="$(sbatch "${args[@]}" "$@" "$sbatch_file")"
  result="${result%%;*}"
  [[ "$result" =~ ^[0-9]+$ ]] || die "sbatch returned invalid job ID: $result"
  printf '%s\n' "$result"
}

register() {
  local label="$1" id="$2" name="$3" parents="$4" submission="$5" entrypoint="$6" receipt="$7"
  graph_jobs+=("$label|$id|$name|$parents|$submission|$entrypoint|$receipt")
}

if [[ "$DRY_RUN" != true ]]; then
  command -v sbatch >/dev/null 2>&1 || die "sbatch is required"
  command -v scontrol >/dev/null 2>&1 || die "scontrol is required"
  command -v scancel >/dev/null 2>&1 || die "scancel is required"
  command -v squeue >/dev/null 2>&1 || die "squeue is required"
  command -v flock >/dev/null 2>&1 || die "flock is required"
  mkdir -p "$JOB_HOME/.cache/jepa4d"
  exec 9>"$JOB_HOME/.cache/jepa4d/slurm-submit.lock"
  flock -x 9 || die "unable to acquire the submission lock"
  active="$(squeue -r -h -u "$(id -un)" -o '%i' | awk 'NF {print $1}' | sort -u)" || die "unable to query active jobs"
  active_count="$(printf '%s\n' "$active" | awk 'NF {count++} END {print count+0}')"
  (( active_count == 0 )) || die "formal Phase 2g requires zero pre-existing active tasks; found $active_count"
fi

JOB_PATH="${PATH:?PATH is required}"
for value in "$ROOT" "$PREFLIGHT" "$GRAPH" "$OUTPUT_ROOT" "$TEST_RECEIPT" "$SUN_ARCHIVE" \
  "$VJEPA_CHECKPOINT" "$VJEPA_IMPLEMENTATION" "$PREREGISTRATION" "$REGISTRY" "$LEDGER" "$READINESS" \
  "$VALIDATION_STATE_ROOT" "$EXECUTION_ID" "$WANDB_ENTITY" "$WANDB_PROJECT" \
  "$JOB_HOME" "$JOB_PATH" "$PYTHON" "$LOG_ROOT"; do
  [[ "$value" != *','* && "$value" != *$'\n'* ]] || die "Slurm export values may not contain commas/newlines"
done
common="HOME=$JOB_HOME,PATH=$JOB_PATH,PYTHONPATH=$ROOT,JEPA4D_PYTHON=$PYTHON,JEPA4D_REPO_ROOT=$ROOT,JEPA4D_EXECUTION_ID=$EXECUTION_ID,JEPA4D_GRAPH=$GRAPH,JEPA4D_PREFLIGHT=$PREFLIGHT,JEPA4D_PREREGISTRATION=$PREREGISTRATION,JEPA4D_TEST_RECEIPT=$TEST_RECEIPT,JEPA4D_OUTPUT_ROOT=$OUTPUT_ROOT,JEPA4D_LOG_ROOT=$LOG_ROOT,JEPA4D_SHORT_COMMIT=$SHORT,JEPA4D_VALIDATION_STATE_ROOT=$VALIDATION_STATE_ROOT,JEPA4D_WANDB_ENTITY=$WANDB_ENTITY,JEPA4D_WANDB_PROJECT=$WANDB_PROJECT,WANDB_MODE=online,PYTHONUTF8=1"
cache_assets="$common,JEPA4D_SUN_ARCHIVE=$SUN_ARCHIVE,JEPA4D_SUN_MATERIALIZATION=$OUTPUT_ROOT/protected-sun-materialization,JEPA4D_VJEPA_CHECKPOINT=$VJEPA_CHECKPOINT,JEPA4D_VJEPA_IMPLEMENTATION=$VJEPA_IMPLEMENTATION"

name="p2gq-T-$SHORT"; T="$(submit "$name" - "$common" slurm/phase2g_tests.sbatch)"; submission_ids+=("$T")
register T "$T" "$name" - slurm/phase2g_tests.sbatch slurm/phase2g_tests.sbatch "$TEST_RECEIPT"

O_OUT="$OUTPUT_ROOT/opacity"; name="p2gq-O-$SHORT"
O="$(submit "$name" "$T" "$common,JEPA4D_STAGE_OUTPUT=$O_OUT" slurm/phase2g_opacity.sbatch)"; submission_ids+=("$O")
register O "$O" "$name" T slurm/phase2g_opacity.sbatch slurm/phase2g_opacity.sbatch "$O_OUT/opacity_receipt.json"

C_OUT="$OUTPUT_ROOT/cache"; name="p2gq-C-$SHORT"
C="$(submit "$name" "$T" "$cache_assets,JEPA4D_STAGE_OUTPUT=$C_OUT" slurm/phase2g_cache.sbatch)"; submission_ids+=("$C")
register C "$C" "$name" T slurm/phase2g_cache.sbatch slurm/phase2g_cache.sbatch "$C_OUT/cache_receipt.json"

Q_OUT="$OUTPUT_ROOT/audit"; name="p2gq-Q-$SHORT"
Q="$(submit "$name" "$O:$C" "$common,JEPA4D_STAGE_OUTPUT=$Q_OUT,JEPA4D_CACHE_ROOT=$C_OUT,JEPA4D_SUN_MATERIALIZATION=$OUTPUT_ROOT/protected-sun-materialization,JEPA4D_OPACITY_RECEIPT=$O_OUT/opacity_receipt.json" slurm/phase2g_audit.sbatch)"; submission_ids+=("$Q")
register Q "$Q" "$name" O,C slurm/phase2g_audit.sbatch slurm/phase2g_audit.sbatch "$Q_OUT/audit_receipt.json"

name="p2gq-H-$SHORT"
H="$(submit "$name" "$Q" "$common,JEPA4D_DISPATCH_STAGE=tuning,JEPA4D_CACHE_ROOT=$C_OUT,JEPA4D_AUDIT_RECEIPT=$Q_OUT/audit_receipt.json" slurm/phase2g_array_dispatch.sbatch --array=0-47%8)"; submission_ids+=("$H")
h_labels=(); task=0
for arm in M0 M1 M2 M3; do for rotation in R0 R1 R2 R3; do for lr in 0 1 2; do
  label="H-$arm-$rotation-L$lr"; h_labels+=("$label")
  register "$label" "${H}_${task}" "p2gq-$label-$SHORT" Q slurm/phase2g_array_dispatch.sbatch slurm/phase2g_tune.sbatch \
    "$OUTPUT_ROOT/tuning/$arm/$rotation/lr-$lr/training_receipt.json"
  task=$((task + 1))
done; done; done

HG_OUT="$OUTPUT_ROOT/lr-selection"; name="p2gq-HG-$SHORT"
HG="$(submit "$name" "$H" "$common,JEPA4D_STAGE_OUTPUT=$HG_OUT,JEPA4D_TUNING_ROOT=$OUTPUT_ROOT/tuning" slurm/phase2g_lr_select.sbatch)"; submission_ids+=("$HG")
h_parents="$(IFS=,; printf '%s' "${h_labels[*]}")"
register HG "$HG" "$name" "$h_parents" slurm/phase2g_lr_select.sbatch slurm/phase2g_lr_select.sbatch "$HG_OUT/lr_selection.json"

name="p2gq-F-$SHORT"
F="$(submit "$name" "$HG" "$common,JEPA4D_DISPATCH_STAGE=formal,JEPA4D_CACHE_ROOT=$C_OUT,JEPA4D_LR_SELECTION=$HG_OUT/lr_selection.json" slurm/phase2g_array_dispatch.sbatch --array=0-47%8)"; submission_ids+=("$F")
f_labels=(); task=0
for arm in M0 M1 M2 M3; do for rotation in R0 R1 R2 R3; do for seed in 0 1 2; do
  label="F-$arm-$rotation-S$seed"; f_labels+=("$label")
  register "$label" "${F}_${task}" "p2gq-$label-$SHORT" HG slurm/phase2g_array_dispatch.sbatch slurm/phase2g_train.sbatch \
    "$OUTPUT_ROOT/formal/$arm/$rotation/seed-$seed/training_receipt.json"
  task=$((task + 1))
done; done; done

name="p2gq-V-$SHORT"
V="$(submit "$name" "$F" "$common,JEPA4D_DISPATCH_STAGE=evaluation,JEPA4D_CACHE_ROOT=$C_OUT" slurm/phase2g_array_dispatch.sbatch --array=0-47%8)"; submission_ids+=("$V")
v_labels=(); task=0
for arm in M0 M1 M2 M3; do for rotation in R0 R1 R2 R3; do for seed in 0 1 2; do
  label="V-$arm-$rotation-S$seed"; parent="F-$arm-$rotation-S$seed"; v_labels+=("$label")
  register "$label" "${V}_${task}" "p2gq-$label-$SHORT" "$parent" slurm/phase2g_array_dispatch.sbatch slurm/phase2g_evaluate.sbatch \
    "$OUTPUT_ROOT/evaluation/$arm/$rotation/seed-$seed/evaluation_receipt.json"
  task=$((task + 1))
done; done; done

S_OUT="$OUTPUT_ROOT/selection"; name="p2gq-S-$SHORT"
S="$(submit "$name" "$V" "$common,JEPA4D_STAGE_OUTPUT=$S_OUT,JEPA4D_EVALUATION_ROOT=$OUTPUT_ROOT/evaluation" slurm/phase2g_select.sbatch)"; submission_ids+=("$S")
v_parents="$(IFS=,; printf '%s' "${v_labels[*]}")"
register S "$S" "$name" "$v_parents" slurm/phase2g_select.sbatch slurm/phase2g_select.sbatch "$S_OUT/selector.json"

G_OUT="$OUTPUT_ROOT/external-seal"; name="p2gq-G-$SHORT"
G="$(submit "$name" "$S" "$common,JEPA4D_STAGE_OUTPUT=$G_OUT,JEPA4D_SELECTOR_RECEIPT=$S_OUT/selector.json" slurm/phase2g_external_guard.sbatch)"; submission_ids+=("$G")
register G "$G" "$name" S slurm/phase2g_external_guard.sbatch slurm/phase2g_external_guard.sbatch "$G_OUT/seal_receipt.json"

Z_OUT="$OUTPUT_ROOT/postflight"; name="p2gq-Z-$SHORT"
Z="$(submit "$name" "$G" "$common,JEPA4D_STAGE_OUTPUT=$Z_OUT,JEPA4D_SEAL_RECEIPT=$G_OUT/seal_receipt.json" slurm/phase2g_postflight.sbatch)"; submission_ids+=("$Z")
register Z "$Z" "$name" G slurm/phase2g_postflight.sbatch slurm/phase2g_postflight.sbatch "$Z_OUT/postflight_receipt.json"

[[ ${#submission_ids[@]} -eq 11 ]] || die "expected 11 base submissions, found ${#submission_ids[@]}"
[[ ${#graph_jobs[@]} -eq 152 ]] || die "expected 152 logical jobs, found ${#graph_jobs[@]}"
graph_args=()
for specification in "${graph_jobs[@]}"; do graph_args+=(--job "$specification"); done
writer_args=(--repo-root "$ROOT" --execution-id "$EXECUTION_ID" --preregistration "$PREREGISTRATION" \
  --preflight "$PREFLIGHT" --registry "$REGISTRY" --ledger "$LEDGER" --readiness "$READINESS" \
  --test-receipt "$TEST_RECEIPT" \
  --output-root "$OUTPUT_ROOT" --source "vjepa_checkpoint=$VJEPA_CHECKPOINT" \
  --source "vjepa_implementation=$VJEPA_IMPLEMENTATION" "${graph_args[@]}")
if [[ "$DRY_RUN" == true ]]; then
  "$PYTHON" scripts/write_phase2g_dependency_graph.py "${writer_args[@]}" --validate-only
  printf 'dry_run=pass\nexecution_id=%s\nbase_submissions=11\nlogical_jobs=152\nsubmitted=false\n' "$EXECUTION_ID"
  released=true
  trap cleanup_temp EXIT
  exit 0
fi
"$PYTHON" scripts/write_phase2g_dependency_graph.py "${writer_args[@]}" --output "$GRAPH"
[[ -s "$GRAPH" ]] || die "canonical Phase 2g dependency graph was not written"
joined="$(IFS=,; printf '%s' "${submission_ids[*]}")"
printf 'release-attempted\n' >"$GATE_ROOT/release-attempted"
scontrol release "$joined"
released=true
trap cleanup_temp EXIT
printf 'execution_id=%s\ngraph=%s\ntest_job=%s\npostflight_job=%s\nbase_submissions=11\nlogical_jobs=152\ncommit=%s\n' \
  "$EXECUTION_ID" "$GRAPH" "$T" "$Z" "$COMMIT"
