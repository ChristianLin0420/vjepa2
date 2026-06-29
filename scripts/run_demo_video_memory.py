"""Deterministic incremental-memory demo with persistence, reload, replay, and reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jepa4d.memory.lod_policy import LODPolicy
from jepa4d.memory.memory_update import FourDMemoryCore
from jepa4d.memory.persistence import MemoryPersistence
from jepa4d.models.object_slot_grounder import ObjectSlot
from jepa4d.planning.query_api import WorldModelQueryAPI
from jepa4d.visualization.experiment_record import ArtifactRecord, ExperimentRecord, PanelRecord, StageRecord
from jepa4d.visualization.memory_report import build_memory_report
from jepa4d.visualization.observability import ExperimentLogger


def observed_slot(step: int) -> ObjectSlot:
    return ObjectSlot(
        object_id="demo-mug-001",
        category="mug",
        description="red ceramic mug",
        pose_map=[1.0 + 0.18 * step, 0.5 + 0.04 * step, 0.8],
        visual_embedding=np.asarray([1.0, 0.2, 0.1], dtype=np.float32),
        affordances={"graspable": 0.9},
        states={"visible": 1.0, "dynamic": 1.0 if step > 0 else 0.0},
        confidence={"overall": 0.55 + 0.04 * step},
        last_seen_time=float(step),
        observation_refs=[f"video-frame-{step:03d}"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/demo_video_memory"))
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-mode", default="online")
    args = parser.parse_args()
    if args.steps < 2:
        raise SystemExit("--steps must be at least 2")
    args.output.mkdir(parents=True, exist_ok=True)
    persistence = MemoryPersistence(args.output / "memory.db")
    memory = FourDMemoryCore()
    logger = ExperimentLogger(
        enabled=args.wandb,
        name="phase4-incremental-memory-demo",
        mode=args.wandb_mode,
        tags=["phase-4", "memory", "persistence", "mock-observations"],
        config={"steps": args.steps, "object_id": "demo-mug-001", "quality_claim": "contract-only"},
    )
    update_results = []
    for step in range(args.steps):
        objects = [] if step == args.steps // 2 else [observed_slot(step)]
        result = memory.update(None, objects, timestamp=float(step), persistence=persistence)
        update_results.append(result.to_serializable())
        logger.log_memory_update(result, memory)
    loaded = FourDMemoryCore.load(persistence)
    replayed = FourDMemoryCore.replay(persistence)
    if loaded.snapshot().to_serializable() != replayed.snapshot().to_serializable():
        raise RuntimeError("snapshot reload and event replay diverged")
    compressed = LODPolicy(max_object_history=4, max_events=4, max_local_observations=4).compress(loaded.snapshot())
    query = WorldModelQueryAPI(loaded)
    query_result = query.find_object("red mug")
    history = query.get_observation_history("demo-mug-001")
    logger.log_memory_snapshot(loaded)
    report_path = build_memory_report(
        loaded, args.output / "report.html", persistence_stats=persistence.stats(), wandb_url=logger.url
    )
    snapshot_path = args.output / "memory.json"
    snapshot_path.write_text(json.dumps(loaded.snapshot().to_serializable(), indent=2) + "\n")
    metrics = {
        "reload_replay_equal": True,
        "revision": loaded.revision,
        "objects": len(loaded.scene_graph.objects),
        "events": len(loaded.episodic_memory.events),
        "history_entries": len(history["observations"]),
        "query_matches": len(query_result),
        "compressed_history_entries": len(compressed.scene_graph.objects["demo-mug-001"].history),
        "persistence": persistence.stats(),
    }
    metrics_path = args.output / "metrics.json"
    metrics_path.write_text(json.dumps({"metrics": metrics, "updates": update_results}, indent=2) + "\n")
    experiment_path = ExperimentRecord(
        title="Phase 4 incremental memory demo",
        experiment_id=f"memory-fixture-{args.steps}u",
        stage="memory",
        status="complete",
        evidence_level="contract-only",
        objective="Validate incremental updates, occlusion, queries, LOD, persistence reload, and event replay.",
        hypothesis="Snapshot reload and event replay remain equal while an empty observation does not create evidence.",
        decision="Promote the memory lifecycle contract, not tracking quality.",
        wandb_url=logger.url,
        config={"steps": args.steps, "fixture": "synthetic moving mug with one occluded step"},
        stages=[
            StageRecord("update", "FourDMemoryCore", "pass", "fixture observations", "revision history"),
            StageRecord("persistence", "SQLite + replay", "pass", "events/snapshots", "equal restored states"),
            StageRecord("query/LOD", "query API + LODPolicy", "pass", "restored memory", "results/compressed history"),
        ],
        panels=[
            PanelRecord("memory/revision", "line", "Verify deterministic state progression."),
            PanelRecord("memory/history_entries", "line", "Inspect evidence accumulation and the occluded step."),
            PanelRecord("memory/mean_confidence", "line", "Inspect confidence through the episode."),
            PanelRecord("memory/objects", "table", "Audit current object records."),
            PanelRecord("memory/events", "table", "Audit episodic changes."),
        ],
        metrics=metrics,
        artifacts=[
            ArtifactRecord(snapshot_path, "JSON", "Serialized snapshot"),
            ArtifactRecord(metrics_path, "JSON", "Machine-readable update metrics"),
            ArtifactRecord(args.output / "memory.db", "SQLite", "Persistent memory"),
            ArtifactRecord(report_path, "HTML", "Interactive lifecycle report"),
        ],
        limitations=["Synthetic observations validate contracts only and do not measure association accuracy."],
        next_actions=["Drive the same lifecycle with real video observations and evaluate identity switches."],
    ).write(args.output / "EXPERIMENT.md")
    for path, artifact_type in (
        (snapshot_path, "memory-snapshot"),
        (metrics_path, "memory-metrics"),
        (args.output / "memory.db", "world-memory"),
        (report_path, "interactive-report"),
        (experiment_path, "experiment-record"),
    ):
        logger.log_artifact(path, artifact_type)
    logger.finish({"result": "success", **metrics})
    print(
        json.dumps(
            {
                "snapshot": str(snapshot_path),
                "database": str(args.output / "memory.db"),
                "metrics": str(metrics_path),
                "report": str(report_path),
                "experiment": str(experiment_path),
                "wandb_url": logger.url,
                "summary": metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
