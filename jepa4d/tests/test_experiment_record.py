from jepa4d.visualization.experiment_record import ExperimentRecord, PanelRecord, StageRecord


def test_experiment_record_is_structured_and_extensible(tmp_path):
    record = ExperimentRecord(
        title="Test run",
        experiment_id="phase|fixture",
        stage="representation",
        status="complete",
        evidence_level="contract-only",
        objective="Validate the record contract.",
        hypothesis="A structured record remains readable and extensible.",
        decision="Promote only as infrastructure evidence.",
        config={"seed": 0},
        metrics={"finite_fraction": 1.0},
        stages=[StageRecord("features", "mock", "pass", "RGB", "tokens", "All outputs finite.")],
        panels=[PanelRecord("features/value_histogram", "histogram", "Detect collapse.", "Finite and non-constant.")],
        extra_sections={"Failure notes": "No failures observed."},
        limitations=["Mock outputs are not model-quality evidence."],
        next_actions=["Run the real checkpoint."],
    )
    output = record.write(tmp_path / "EXPERIMENT.md")
    text = output.read_text()

    assert "phase\\|fixture" in text
    assert "## Stage results and insights" in text
    assert "## W&B dashboard reading guide" in text
    assert "## Failure notes" in text
    assert '"seed": 0' in text
    assert text.endswith("\n")
