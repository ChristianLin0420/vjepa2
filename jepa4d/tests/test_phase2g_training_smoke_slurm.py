from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SBATCH = ROOT / "slurm" / "phase2g_training_smoke.sbatch"
SUBMITTER = ROOT / "slurm" / "submit_phase2g_training_smoke.sh"
ACCOUNT = "edgeai_tao-ptm_image-foundation-model-clip"
PARTITIONS = "polar4,polar3,polar,batch_block1,grizzly,batch_block2,batch_block3"


def _directives(source: str) -> dict[str, str]:
    return dict(re.findall(r"^#SBATCH --([^=]+)=(.+)$", source, flags=re.MULTILINE))


def _fixture_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    (repo / "slurm").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / ".conda-gpu" / "bin").mkdir(parents=True)
    shutil.copy2(SUBMITTER, repo / "slurm" / SUBMITTER.name)
    shutil.copy2(SBATCH, repo / "slurm" / SBATCH.name)
    shutil.copy2(ROOT / ".gitignore", repo / ".gitignore")
    (repo / "scripts" / "run_phase2g_training_smoke.py").write_text("# fixture\n", encoding="utf-8")
    python = repo / ".conda-gpu" / "bin" / "python"
    python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    python.chmod(0o755)
    subprocess.run(("git", "init", "-q", str(repo)), check=True)
    subprocess.run(("git", "-C", str(repo), "add", "."), check=True)
    subprocess.run(
        (
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Slurm Test",
            "-c",
            "user.email=slurm-test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ),
        check=True,
    )

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    squeue = fake_bin / "squeue"
    squeue.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$*\" == *'%i'* ]]; then\n"
        "  printf '%s' \"${FAKE_ACTIVE_JOB_TASKS:-}\"\n"
        "else\n"
        "  printf '%s' \"${FAKE_QUEUED_NAMES:-}\"\n"
        "fi\n",
        encoding="utf-8",
    )
    sbatch = fake_bin / "sbatch"
    sbatch.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" >\"$FAKE_SBATCH_CAPTURE\"\nprintf '12345;fixture-cluster\\n'\n",
        encoding="utf-8",
    )
    squeue.chmod(0o755)
    sbatch.chmod(0o755)
    return repo, fake_bin


def _submission_environment(repo: Path, fake_bin: Path, tmp_path: Path) -> dict[str, str]:
    home = tmp_path / "home"
    home.mkdir()
    netrc = home / ".netrc"
    netrc.write_text("machine api.wandb.ai login fixture password fixture\n", encoding="utf-8")
    netrc.chmod(0o600)
    return {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(home),
        "JEPA4D_STAGE_OUTPUT": str(repo / "outputs" / "phase2g-training-smoke" / "fixture-execution"),
        "JEPA4D_EXECUTION_ID": "fixture-execution",
        "JEPA4D_RUN_NAME": "fixture-run",
        "JEPA4D_JOB_NAME": "j4d-p2g-smoke-fixture",
        "JEPA4D_MAX_STEPS": "3",
        "JEPA4D_WANDB_PROJECT": "fixture-project",
        "JEPA4D_WANDB_ENTITY": "fixture-entity",
        "FAKE_SBATCH_CAPTURE": str(tmp_path / "sbatch-arguments.txt"),
    }


def _run_submitter(repo: Path, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("bash", str(repo / "slurm" / SUBMITTER.name)),
        cwd=repo,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_sbatch_is_one_short_gpu_allocation_with_exact_runner_contract() -> None:
    source = SBATCH.read_text(encoding="utf-8")
    directives = _directives(source)
    assert directives["job-name"].startswith("j4d-p2g-smoke-")
    assert directives["account"] == ACCOUNT
    assert directives["partition"] == PARTITIONS
    assert directives["nodes"] == "1"
    assert directives["ntasks"] == "1"
    assert directives["gres"] == "gpu:1"
    hours, minutes, seconds = (int(value) for value in directives["time"].split(":"))
    assert hours * 3600 + minutes * 60 + seconds <= 30 * 60
    assert "--array" not in source
    assert 'scontrol show job -o "$JOB_ID"' in source
    for allocation_field in ("Account", "Partition", "NumNodes", "NumTasks", "AllocTRES", "TimeLimit"):
        assert allocation_field in source
    assert "time_seconds <= 30 * 60" in source
    assert "ArrayJobId" in source and "ArrayTaskId" in source
    assert "torch.cuda.device_count() != 1" in source
    assert '"$REPO_ROOT/scripts/run_phase2g_training_smoke.py"' in source
    for argument in (
        "--output",
        "--max-steps",
        "--device cuda:0",
        "--execution-id",
        "--run-name",
        "--wandb-project",
        "--wandb-entity",
    ):
        assert argument in source
    assert "WANDB_MODE=online" in source
    assert "JEPA4D_GPU_MONITOR_INTERVAL=1" in source
    assert 'jepa4d_start_gpu_monitor "$JEPA4D_JOB_LOG_DIR/gpu-telemetry.csv"' in source


def test_sbatch_requires_complete_output_contract_before_success() -> None:
    source = SBATCH.read_text(encoding="utf-8")
    runner_position = source.index("run_phase2g_training_smoke.py")
    success_position = source.rindex("printf 'pass\\n' >\"$JEPA4D_JOB_LOG_DIR/SUCCESS\"")
    assert runner_position < success_position
    for artifact in (
        "training_receipt.json",
        "steps.jsonl",
        "checkpoints/M0.pt",
        "checkpoints/M1.pt",
        "checkpoints/M2.pt",
        "checkpoints/M3.pt",
        "wandb_receipt.json",
        "SUCCESS",
    ):
        assert artifact in source
        assert source.index(artifact) < success_position
    assert '[[ -f "$artifact" && ! -L "$artifact" && -s "$artifact" ]]' in source


def test_submitter_has_clean_fresh_safe_and_fail_closed_gates() -> None:
    source = SUBMITTER.read_text(encoding="utf-8")
    assert os.access(SUBMITTER, os.X_OK)
    assert "git rev-parse --verify 'HEAD^{commit}'" in source
    assert "git status --porcelain=v1 --untracked-files=all" in source
    assert '[[ ! -e "$STAGE_OUTPUT" && ! -L "$STAGE_OUTPUT" ]]' in source
    assert 'DEFAULT_SUFFIX="${SHORT}-${STAMP}-${NONCE}"' in source
    assert "execution, W&B run, and Slurm job names must be distinct" in source
    assert "reject_sensitive_identifier JEPA4D_JOB_NAME" in source
    assert '[[ "$MAX_STEPS" =~ ^([1-9]|10)$ ]]' in source
    assert 'exec 9>"$LOCK_ROOT/slurm-submit.lock"' in source
    assert 'squeue -r -h -u "$SCHEDULER_USER" -o "%i"' in source
    assert "ACTIVE_STATES" not in source
    assert "sort -u" in source
    assert "${#active_job_tasks[@]} >= 8" in source
    assert "flock -x 9" in source
    assert 'squeue -h -u "$SCHEDULER_USER" -o "%j"' in source
    assert "Slurm job name is already present" in source
    assert source.index("squeue -r -h") < source.index("sbatch --parsable")
    assert source.count("sbatch --parsable") == 1


def test_submitter_uses_fake_scheduler_and_never_exports_credentials(tmp_path: Path) -> None:
    repo, fake_bin = _fixture_repo(tmp_path)
    environment = _submission_environment(repo, fake_bin, tmp_path)
    environment["FAKE_ACTIVE_JOB_TASKS"] = "101\n101\n202_1\n202_2\n303\n404\n505\n606\n"
    environment["WANDB_API_KEY"] = "must-not-cross-the-submission-boundary"
    environment["HF_TOKEN"] = "must-not-cross-either"
    result = _run_submitter(repo, environment)
    assert result.returncode == 0, result.stderr
    parsed = dict(line.split("=", 1) for line in result.stdout.splitlines())
    assert parsed == {
        "job_id": "12345",
        "output_path": environment["JEPA4D_STAGE_OUTPUT"],
        "stdout_log": f"{repo}/outputs/slurm-submit-logs/j4d-p2g-smoke-fixture-12345.out",
        "stderr_log": f"{repo}/outputs/slurm-submit-logs/j4d-p2g-smoke-fixture-12345.err",
        "structured_log_dir": (f"{repo}/outputs/slurm_logs/phase2g-training-smoke/j4d-p2g-smoke-fixture-12345"),
        "active_job_tasks_before_submission": "7",
    }
    arguments = Path(environment["FAKE_SBATCH_CAPTURE"]).read_text(encoding="utf-8")
    assert "--parsable\n--job-name\nj4d-p2g-smoke-fixture\n" in arguments
    assert f"--output\n{repo}/outputs/slurm-submit-logs/%x-%j.out\n" in arguments
    assert f"--error\n{repo}/outputs/slurm-submit-logs/%x-%j.err\n" in arguments
    assert "--export\n" in arguments
    assert "JEPA4D_MAX_STEPS=3" in arguments
    assert "JEPA4D_GPU_MONITOR_INTERVAL=1" in arguments
    assert "WANDB_MODE=online" in arguments
    assert "must-not-cross-the-submission-boundary" not in arguments
    assert "must-not-cross-either" not in arguments
    assert "WANDB_API_KEY" not in arguments
    assert "HF_TOKEN" not in arguments
    assert "ALL," not in arguments
    assert not arguments.split("--export\n", 1)[1].startswith("ALL")
    assert not Path(environment["JEPA4D_STAGE_OUTPUT"]).exists()
    assert not subprocess.check_output(
        ("git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=all"), text=True
    ).strip()


@pytest.mark.parametrize("active_count", (8, 9))
def test_submitter_refuses_eight_or_more_expanded_active_tasks(tmp_path: Path, active_count: int) -> None:
    repo, fake_bin = _fixture_repo(tmp_path)
    environment = _submission_environment(repo, fake_bin, tmp_path)
    environment["FAKE_ACTIVE_JOB_TASKS"] = "\n".join(str(100 + index) for index in range(active_count)) + "\n"
    result = _run_submitter(repo, environment)
    assert result.returncode == 2
    assert f"found {active_count} distinct active job tasks" in result.stderr
    assert not Path(environment["FAKE_SBATCH_CAPTURE"]).exists()


@pytest.mark.parametrize("max_steps", ("0", "11", "1.5", "three"))
def test_submitter_rejects_invalid_step_bound(tmp_path: Path, max_steps: str) -> None:
    repo, fake_bin = _fixture_repo(tmp_path)
    environment = _submission_environment(repo, fake_bin, tmp_path)
    environment["JEPA4D_MAX_STEPS"] = max_steps
    result = _run_submitter(repo, environment)
    assert result.returncode == 2
    assert "must be an integer from 1 through 10" in result.stderr
    assert not Path(environment["FAKE_SBATCH_CAPTURE"]).exists()


def test_submitter_requires_mode_0600_home_auth(tmp_path: Path) -> None:
    repo, fake_bin = _fixture_repo(tmp_path)
    environment = _submission_environment(repo, fake_bin, tmp_path)
    (Path(environment["HOME"]) / ".netrc").chmod(0o640)
    result = _run_submitter(repo, environment)
    assert result.returncode == 2
    assert "HOME/.netrc must have mode 0600" in result.stderr
    assert not Path(environment["FAKE_SBATCH_CAPTURE"]).exists()


def test_slurm_sources_have_no_data_or_model_inputs() -> None:
    source = (SBATCH.read_text(encoding="utf-8") + SUBMITTER.read_text(encoding="utf-8")).casefold()
    forbidden = (
        "dataset_root",
        "dataset-root",
        "archive_path",
        "--archive",
        "model_id",
        "model-id",
        "checkpoint_path",
        "--checkpoint",
    )
    assert all(marker not in source for marker in forbidden)
