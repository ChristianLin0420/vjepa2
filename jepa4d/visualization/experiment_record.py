"""Structured, extensible Markdown records for JEPA-4D experiments."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _cell(value: Any) -> str:
    """Render a value safely inside a Markdown table cell."""
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, sort_keys=True, default=str)
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _git_state() -> tuple[str, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True, text=True).stdout
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", None


@dataclass(frozen=True)
class StageRecord:
    """One pipeline stage evaluated by a run."""

    name: str
    implementation: str
    status: str
    inputs: str
    outputs: str
    insight: str = "Pending review."


@dataclass(frozen=True)
class PanelRecord:
    """A W&B panel and the question it is intended to answer."""

    path: str
    kind: str
    purpose: str
    interpretation: str = "Inspect alongside the numerical artifact."


@dataclass(frozen=True)
class ArtifactRecord:
    path: str | Path
    kind: str
    purpose: str


@dataclass
class ExperimentRecord:
    """Canonical experiment narrative with optional domain-specific extensions."""

    title: str
    experiment_id: str
    stage: str
    status: str
    objective: str
    hypothesis: str
    decision: str
    evidence_level: str = "integration"
    wandb_url: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    stages: list[StageRecord] = field(default_factory=list)
    panels: list[PanelRecord] = field(default_factory=list)
    artifacts: list[ArtifactRecord] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    extra_sections: dict[str, str] = field(default_factory=dict)

    def to_markdown(self) -> str:
        commit, dirty = _git_state()
        lines = [
            f"# {self.title}",
            "",
            "## Experiment metadata",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Experiment ID | `{_cell(self.experiment_id)}` |",
            f"| Stage | `{_cell(self.stage)}` |",
            f"| Status | `{_cell(self.status)}` |",
            f"| Evidence level | `{_cell(self.evidence_level)}` |",
            f"| Timestamp | `{_cell(self.timestamp)}` |",
            f"| Git commit | `{commit}` |",
            f"| Dirty worktree | `{dirty if dirty is not None else 'unknown'}` |",
            f"| W&B | {self.wandb_url or 'disabled'} |",
            "",
            "## Question and decision",
            "",
            f"- Objective: {self.objective}",
            f"- Hypothesis: {self.hypothesis}",
            f"- Decision: {self.decision}",
        ]
        if self.stages:
            lines += [
                "",
                "## Stage results and insights",
                "",
                "| Stage | Implementation | Status | Inputs | Outputs | Insight / decision |",
                "|---|---|---|---|---|---|",
            ]
            lines += [
                f"| {_cell(s.name)} | {_cell(s.implementation)} | {_cell(s.status)} | {_cell(s.inputs)} | "
                f"{_cell(s.outputs)} | {_cell(s.insight)} |"
                for s in self.stages
            ]
        if self.config:
            lines += [
                "",
                "## Reproduction configuration",
                "",
                "```json",
                json.dumps(self.config, indent=2, default=str),
                "```",
            ]
        if self.panels:
            lines += [
                "",
                "## W&B dashboard reading guide",
                "",
                "| Panel / namespace | Type | What it answers | How to interpret this run |",
                "|---|---|---|---|",
            ]
            lines += [
                f"| `{_cell(p.path)}` | {_cell(p.kind)} | {_cell(p.purpose)} | {_cell(p.interpretation)} |"
                for p in self.panels
            ]
        if self.metrics:
            lines += ["", "## Numerical results", "", "| Metric | Value |", "|---|---|"]
            lines += [f"| `{_cell(key)}` | `{_cell(value)}` |" for key, value in self.metrics.items()]
        if self.artifacts:
            lines += ["", "## Artifacts", "", "| Path | Type | Purpose |", "|---|---|---|"]
            lines += [f"| `{_cell(a.path)}` | {_cell(a.kind)} | {_cell(a.purpose)} |" for a in self.artifacts]
        for heading, body in self.extra_sections.items():
            lines += ["", f"## {heading}", "", body.rstrip()]
        if self.limitations:
            lines += ["", "## Claim boundary and limitations", ""] + [f"- {item}" for item in self.limitations]
        if self.next_actions:
            lines += ["", "## Next experiments", ""] + [f"- {item}" for item in self.next_actions]
        lines += [""]
        return "\n".join(lines)

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_markdown())
        return target
