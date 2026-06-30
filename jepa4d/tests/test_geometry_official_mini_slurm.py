from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SBATCH = ROOT / "slurm" / "geometry_official_mini.sbatch"
SUBMITTER = ROOT / "slurm" / "submit_geometry_official_mini.sh"
ACCOUNT = "edgeai_tao-ptm_image-foundation-model-clip"
PARTITIONS = "polar4,polar3,polar,batch_block1,grizzly,batch_block2,batch_block3"


def _directives(source: str) -> dict[str, str]:
    return dict(re.findall(r"^#SBATCH --([^=]+)=(.+)$", source, flags=re.MULTILINE))


def _fixture_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    (repo / "slurm").mkdir(parents=True)
    (repo / "configs" / "validation").mkdir(parents=True)
    (repo / ".conda-gpu" / "bin").mkdir(parents=True)
    shutil.copy2(SUBMITTER, repo / "slurm" / SUBMITTER.name)
    shutil.copy2(SBATCH, repo / "slurm" / SBATCH.name)
    shutil.copy2(ROOT / ".gitignore", repo / ".gitignore")
    (repo / "configs" / "validation" / "dataset_registry.yaml").write_text("schema: fixture\n")
    (repo / "configs" / "validation" / "consumed_test_ledger.yaml").write_text("schema: fixture\n")
    python = repo / ".conda-gpu" / "bin" / "python"
    python.write_text("#!/usr/bin/env bash\nexit 0\n")
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
        "fi\n"
    )
    sbatch = fake_bin / "sbatch"
    sbatch.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" >\"$FAKE_SBATCH_CAPTURE\"\nprintf '12345;fixture-cluster\\n'\n"
    )
    squeue.chmod(0o755)
    sbatch.chmod(0o755)
    return repo, fake_bin


def _submission_environment(repo: Path, fake_bin: Path, tmp_path: Path) -> dict[str, str]:
    model = tmp_path / "vggt-model"
    model.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    return {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "USER": "slurm-test",
        "HOME": str(home),
        "JEPA4D_TUM_ARCHIVE": str(tmp_path / "must-not-stat-archive.tgz"),
        "JEPA4D_VGGT_CHECKPOINT": str(model),
        "JEPA4D_STAGE_OUTPUT": str(tmp_path / "fresh-output"),
        "JEPA4D_VALIDATION_STATE_ROOT": str(tmp_path / "validation-state"),
        "JEPA4D_EXECUTION_ID": "fixture-execution",
        "JEPA4D_RUN_NAME": "fixture-run",
        "JEPA4D_JOB_NAME": "j4d-gmini-fixture",
        "FAKE_SBATCH_CAPTURE": str(tmp_path / "sbatch-arguments.txt"),
    }


def test_sbatch_is_exactly_one_bounded_gpu_allocation() -> None:
    source = SBATCH.read_text(encoding="utf-8")
    directives = _directives(source)
    assert directives["job-name"].startswith("j4d-gmini-")
    assert directives["account"] == ACCOUNT
    assert directives["partition"] == PARTITIONS
    assert directives["nodes"] == "1"
    assert directives["ntasks"] == "1"
    assert directives["gres"] == "gpu:1"
    hours, minutes, seconds = (int(value) for value in directives["time"].split(":"))
    assert hours * 3600 + minutes * 60 + seconds <= 4 * 3600
    assert "--array" not in source
    assert "-m jepa4d.validation.geometry_official_mini" in source
    validator = "slurm/validate_geometry_official_mini.py"
    runner = "-m jepa4d.validation.geometry_official_mini"
    assert source.index(validator) < source.index("--allocation-only") < source.index(runner)
    assert source.index(runner) < source.rindex(validator)
    assert "WANDB_MODE=online" in source
    assert "JEPA4D_VALIDATION_STATE_ROOT" in source
    assert "JEPA4D_GIT_COMMIT" in source
    assert "repository HEAD changed after submission" in source
    assert 'jepa4d_require_file "$ARCHIVE"' not in source
    assert "JEPA4D_TUM_DATASET_ROOT" not in source


def test_submitter_has_clean_fresh_unique_and_fail_closed_scheduler_gates() -> None:
    source = SUBMITTER.read_text(encoding="utf-8")
    assert os.access(SUBMITTER, os.X_OK)
    assert "git rev-parse --verify 'HEAD^{commit}'" in source
    assert "git status --porcelain=v1 --untracked-files=all" in source
    assert '[[ ! -e "$STAGE_OUTPUT" ]]' in source
    assert '[[ -f "$ARCHIVE" ]]' not in source
    assert "JEPA4D_TUM_DATASET_ROOT" not in source
    assert 'ACTIVE_STATES="PENDING,RUNNING,CONFIGURING,COMPLETING,SUSPENDED"' in source
    assert 'SCHEDULER_USER="$(id -un)"' in source
    assert 'LOCK_ROOT="$JOB_HOME/.cache/jepa4d"' in source
    assert "XDG_RUNTIME_DIR" not in source
    assert 'squeue -r -h -u "$SCHEDULER_USER" -t "$ACTIVE_STATES" -o "%i"' in source
    assert "flock -x 9" in source
    assert "sort -u" in source
    assert "${#active_job_tasks[@]} >= 8" in source
    assert "unable to query active jobs; refusing submission" in source
    assert 'squeue -h -u "$SCHEDULER_USER" -o "%j"' in source
    assert "Slurm job name is already present" in source
    assert 'DEFAULT_SUFFIX="${SHORT}-${STAMP}-${NONCE}"' in source
    assert "execution, W&B run, and Slurm job names must be distinct" in source
    assert "JEPA4D_JOB_NAME must begin with j4d-gmini-" in source
    assert "reject_sensitive_identifier JEPA4D_JOB_NAME" in source
    assert 'SUBMISSION_LOG_ROOT="$ROOT/outputs/slurm-submit-logs"' in source
    assert source.index("squeue -r -h") < source.index("sbatch --parsable")
    assert source.count("sbatch --parsable") == 1
    for token in ("di" + "ode", "s" + "un"):
        assert token not in (SBATCH.read_text(encoding="utf-8") + source).casefold()


def test_submitter_uses_stubbed_scheduler_and_explicit_export_allowlist(tmp_path: Path) -> None:
    repo, fake_bin = _fixture_repo(tmp_path)
    environment = _submission_environment(repo, fake_bin, tmp_path)
    environment["FAKE_ACTIVE_JOB_TASKS"] = "101\n101\n202_1\n202_2\n303\n404\n505\n606\n"
    environment["WANDB_API_KEY"] = "must-not-cross-the-submission-boundary"
    result = subprocess.run(
        ("bash", str(repo / "slurm" / SUBMITTER.name)),
        cwd=repo,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Governed TUM official-mini job: 12345" in result.stdout
    assert "Active job tasks before submission: 7" in result.stdout
    arguments = Path(environment["FAKE_SBATCH_CAPTURE"]).read_text(encoding="utf-8")
    assert "--parsable\n--job-name\nj4d-gmini-fixture\n" in arguments
    assert f"--output\n{repo}/outputs/slurm-submit-logs/%x-%j.out\n" in arguments
    assert f"--error\n{repo}/outputs/slurm-submit-logs/%x-%j.err\n" in arguments
    assert "--export\n" in arguments
    assert "JEPA4D_TUM_ARCHIVE=" in arguments
    assert "JEPA4D_GIT_COMMIT=" in arguments
    assert "JEPA4D_VALIDATION_STATE_ROOT=" in arguments
    assert "WANDB_MODE=online" in arguments
    assert "must-not-cross-the-submission-boundary" not in arguments
    assert "ALL," not in arguments
    assert not Path(environment["JEPA4D_STAGE_OUTPUT"]).exists()
    log_root = repo / "outputs" / "slurm-submit-logs"
    (log_root / "j4d-gmini-fixture-12345.out").write_text("allocated\n", encoding="utf-8")
    (log_root / "j4d-gmini-fixture-12345.err").write_text("", encoding="utf-8")
    assert not subprocess.check_output(
        ("git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=all"), text=True
    ).strip()


@pytest.mark.parametrize(
    "environment_name",
    (
        "JEPA4D_EXECUTION_ID",
        "JEPA4D_RUN_NAME",
        "JEPA4D_JOB_NAME",
        "JEPA4D_WANDB_PROJECT",
        "JEPA4D_WANDB_ENTITY",
    ),
)
def test_submitter_rejects_credential_shaped_public_identifiers_before_sbatch(
    tmp_path: Path, environment_name: str
) -> None:
    repo, fake_bin = _fixture_repo(tmp_path)
    environment = _submission_environment(repo, fake_bin, tmp_path)
    synthetic = "hf_abcdefghijklmnopqrstuvwxyz123456"
    environment[environment_name] = f"j4d-gmini-{synthetic}" if environment_name == "JEPA4D_JOB_NAME" else synthetic
    result = subprocess.run(
        ("bash", str(repo / "slurm" / SUBMITTER.name)),
        cwd=repo,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "resembles credential material" in result.stderr
    assert not Path(environment["FAKE_SBATCH_CAPTURE"]).exists()


@pytest.mark.parametrize("duplicate", (False, True))
def test_submitter_refuses_eight_distinct_active_job_tasks(tmp_path: Path, duplicate: bool) -> None:
    repo, fake_bin = _fixture_repo(tmp_path)
    environment = _submission_environment(repo, fake_bin, tmp_path)
    allocations = [str(value) for value in range(100, 108)]
    if duplicate:
        allocations.extend(("100", "103", "107"))
    environment["FAKE_ACTIVE_JOB_TASKS"] = "\n".join(allocations) + "\n"
    result = subprocess.run(
        ("bash", str(repo / "slurm" / SUBMITTER.name)),
        cwd=repo,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "8 distinct active job tasks" in result.stderr
    assert not Path(environment["FAKE_SBATCH_CAPTURE"]).exists()
