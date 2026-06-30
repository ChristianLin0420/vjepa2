"""Validate, authorize, freeze, and consume registered evaluation targets."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Annotated

import typer

from jepa4d.validation.access import (
    DatasetAccessController,
    SealedTargetAuthorization,
    SignedSealedTargetSelectorReceipt,
    load_ed25519_private_key_from_env,
    write_sealed_target_authorization,
    write_sealed_target_selector_receipt,
)
from jepa4d.validation.ledger import (
    ConsumedTestLedger,
    EventStoreUnavailable,
    FirstOpenLineage,
    FutureUse,
    TimePrecision,
    append_first_open,
    effective_targets,
    freeze_ledger,
    load_events,
)
from jepa4d.validation.registry import AccessOperation, DatasetRegistry, freeze_registry
from jepa4d.validation.split_manifest import SplitManifest

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _print(value: object) -> None:
    typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))


def _atomic_write_text(path: Path, value: str) -> None:
    """Durably replace a text artifact without exposing a partial schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        os.chmod(path, 0o640)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


@app.command("validate")
def validate_command(
    registry_path: Annotated[Path, typer.Option("--registry", exists=True, dir_okay=False)],
    ledger_path: Annotated[Path | None, typer.Option("--ledger", exists=True, dir_okay=False)] = None,
) -> None:
    registry = DatasetRegistry.load(registry_path)
    blocked = {
        dataset.dataset_id: dataset.readiness_blockers for dataset in registry.datasets if dataset.readiness_blockers
    }
    value: dict[str, object] = {
        "status": "schema-valid-audit-only",
        "schema_status": "valid",
        "portfolio_operational_status": "not-evaluated-without-ledger",
        "schema_version": registry.schema_version,
        "registry_sha256": registry.sha256,
        "datasets": len(registry.datasets),
        "splits": sum(len(dataset.splits) for dataset in registry.datasets),
        "dataset_audit_blockers": blocked,
        "audit_ready_datasets": sorted(
            dataset.dataset_id for dataset in registry.datasets if not dataset.readiness_blockers
        ),
        "readiness_semantics": "Audit readiness is not authorization for any operation; use the ledger-aware controller.",
    }
    if ledger_path is not None:
        ledger = ConsumedTestLedger.load(ledger_path)
        ledger.validate_against(registry)
        event_store_error = None
        resolved_instance_sha256 = None
        try:
            resolved_instance_sha256 = ledger.event_store.resolved_instance_sha256
            events = load_events(registry=registry, ledger=ledger)
            targets = effective_targets(registry, ledger, events)
        except EventStoreUnavailable as error:
            events = ()
            targets = effective_targets(registry, ledger, ())
            event_store_error = str(error)
        operational_blockers = list(blocked)
        if event_store_error:
            operational_blockers.append("canonical-event-store-unavailable")
        if not ledger.event_store.externally_append_only:
            operational_blockers.append("external-append-only-durability-not-deployed")
        value.update(
            {
                "ledger_sha256": ledger.sha256,
                "ledger_targets": len(ledger.targets),
                "consumption_events": len(events),
                "effective_consumed_targets": sum(target.state.value == "consumed" for target in targets.values()),
                "canonical_event_store": {
                    "root_env": ledger.event_store.root_env,
                    "relative_path": ledger.event_store.relative_path,
                    "identity_sha256": ledger.event_store.sha256,
                    "resolved_instance_sha256": resolved_instance_sha256,
                    "durability": ledger.event_store.durability,
                    "externally_append_only": ledger.event_store.externally_append_only,
                    "error": event_store_error,
                },
                "portfolio_operational_status": "blocked" if operational_blockers else "ready",
                "operational_blockers": sorted(operational_blockers),
                "status": "schema-valid-but-operationally-blocked" if operational_blockers else "operationally-ready",
            }
        )
    _print(value)


@app.command("authorize")
def authorize_command(
    registry_path: Annotated[Path, typer.Option("--registry", exists=True, dir_okay=False)],
    ledger_path: Annotated[Path, typer.Option("--ledger", exists=True, dir_okay=False)],
    dataset_id: Annotated[str, typer.Option("--dataset")],
    split_id: Annotated[str, typer.Option("--split")],
    operation: Annotated[AccessOperation, typer.Option("--operation")],
    sealed_authorization: Annotated[
        Path | None, typer.Option("--sealed-authorization", exists=True, dir_okay=False)
    ] = None,
) -> None:
    registry = DatasetRegistry.load(registry_path)
    ledger = ConsumedTestLedger.load(ledger_path)
    decision = DatasetAccessController(registry=registry, ledger=ledger).authorize(
        dataset_id,
        split_id,
        operation,
        sealed_authorization=sealed_authorization,
    )
    _print(decision.model_dump(mode="json"))


@app.command("freeze")
def freeze_command(
    registry_path: Annotated[Path, typer.Option("--registry", exists=True, dir_okay=False)],
    ledger_path: Annotated[Path, typer.Option("--ledger", exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option("--output", file_okay=False)],
) -> None:
    registry = DatasetRegistry.load(registry_path)
    ledger = ConsumedTestLedger.load(ledger_path)
    ledger.validate_against(registry)
    registry_snapshot, registry_receipt = freeze_registry(registry_path, output_dir)
    ledger_snapshot = freeze_ledger(registry, ledger, output_dir)
    _print(
        {
            "registry_snapshot": str(registry_snapshot.path),
            "registry_snapshot_sha256": registry_snapshot.sha256,
            "registry_receipt": str(registry_receipt.path),
            "registry_receipt_sha256": registry_receipt.sha256,
            "ledger_snapshot": str(ledger_snapshot.path),
            "ledger_snapshot_sha256": ledger_snapshot.sha256,
        }
    )


@app.command("consume")
def consume_command(
    registry_path: Annotated[Path, typer.Option("--registry", exists=True, dir_okay=False)],
    ledger_path: Annotated[Path, typer.Option("--ledger", exists=True, dir_okay=False)],
    dataset_id: Annotated[str, typer.Option("--dataset")],
    split_id: Annotated[str, typer.Option("--split")],
    operation: Annotated[AccessOperation, typer.Option("--operation")],
    opened_at: Annotated[str, typer.Option("--opened-at")],
    time_precision: Annotated[TimePrecision, typer.Option("--time-precision")],
    experiment_id: Annotated[str, typer.Option("--experiment-id")],
    git_commit: Annotated[str, typer.Option("--git-commit")],
    source_record: Annotated[str, typer.Option("--source-record")],
    future_use: Annotated[list[FutureUse], typer.Option("--future-use")],
    execution_id: Annotated[str | None, typer.Option("--execution-id")] = None,
    source_record_sha256: Annotated[str | None, typer.Option("--source-record-sha256")] = None,
    sealed_authorization: Annotated[
        Path | None, typer.Option("--sealed-authorization", exists=True, dir_okay=False)
    ] = None,
    notes: Annotated[str, typer.Option("--notes")] = "First target opening recorded by validation registry CLI.",
) -> None:
    registry = DatasetRegistry.load(registry_path)
    ledger = ConsumedTestLedger.load(ledger_path)
    artifact = append_first_open(
        registry=registry,
        ledger=ledger,
        dataset_id=dataset_id,
        split_id=split_id,
        operation=operation,
        lineage=FirstOpenLineage(
            opened_at=opened_at,
            time_precision=time_precision,
            experiment_id=experiment_id,
            git_commit=git_commit,
            execution_id=execution_id,
            source_record=source_record,
            source_record_sha256=source_record_sha256,
            notes=notes,
        ),
        permitted_future_uses=frozenset(future_use),
        sealed_authorization=sealed_authorization,
    )
    _print({"event": str(artifact.path), "sha256": artifact.sha256, "bytes": artifact.bytes})


@app.command("issue-sealed-authorization")
def issue_sealed_authorization_command(
    registry_path: Annotated[Path, typer.Option("--registry", exists=True, dir_okay=False)],
    ledger_path: Annotated[Path, typer.Option("--ledger", exists=True, dir_okay=False)],
    dataset_id: Annotated[str, typer.Option("--dataset")],
    split_id: Annotated[str, typer.Option("--split")],
    survivor: Annotated[str, typer.Option("--survivor")],
    preregistration: Annotated[Path, typer.Option("--preregistration", exists=True, dir_okay=False)],
    selector: Annotated[Path, typer.Option("--selector", exists=True, dir_okay=False)],
    checkpoint: Annotated[Path, typer.Option("--checkpoint", exists=True, dir_okay=False)],
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    calibrator: Annotated[Path, typer.Option("--calibrator", exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option("--output", file_okay=False)],
) -> None:
    registry = DatasetRegistry.load(registry_path)
    ledger = ConsumedTestLedger.load(ledger_path)
    artifact = write_sealed_target_authorization(
        registry=registry,
        ledger=ledger,
        dataset_id=dataset_id,
        split_id=split_id,
        survivor=survivor,
        preregistration=preregistration,
        selector=selector,
        checkpoint=checkpoint,
        config=config,
        calibrator=calibrator,
        output_dir=output_dir,
    )
    _print({"authorization": str(artifact.path), "sha256": artifact.sha256, "bytes": artifact.bytes})


@app.command("issue-sealed-selector")
def issue_sealed_selector_command(
    registry_path: Annotated[Path, typer.Option("--registry", exists=True, dir_okay=False)],
    ledger_path: Annotated[Path, typer.Option("--ledger", exists=True, dir_okay=False)],
    dataset_id: Annotated[str, typer.Option("--dataset")],
    split_id: Annotated[str, typer.Option("--split")],
    survivor: Annotated[str, typer.Option("--survivor")],
    git_commit: Annotated[str, typer.Option("--git-commit")],
    preregistration: Annotated[Path, typer.Option("--preregistration", exists=True, dir_okay=False)],
    checkpoint: Annotated[Path, typer.Option("--checkpoint", exists=True, dir_okay=False)],
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    calibrator: Annotated[Path, typer.Option("--calibrator", exists=True, dir_okay=False)],
    private_key_env: Annotated[str, typer.Option("--private-key-env")],
    output_dir: Annotated[Path, typer.Option("--output", file_okay=False)],
) -> None:
    registry = DatasetRegistry.load(registry_path)
    ledger = ConsumedTestLedger.load(ledger_path)
    artifact = write_sealed_target_selector_receipt(
        registry=registry,
        ledger=ledger,
        dataset_id=dataset_id,
        split_id=split_id,
        survivor=survivor,
        git_commit=git_commit,
        preregistration=preregistration,
        checkpoint=checkpoint,
        config=config,
        calibrator=calibrator,
        private_key=load_ed25519_private_key_from_env(private_key_env),
        output_dir=output_dir,
    )
    _print({"selector": str(artifact.path), "sha256": artifact.sha256, "bytes": artifact.bytes})


@app.command("schema")
def schema_command(
    output_dir: Annotated[Path, typer.Option("--output", file_okay=False)],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = output_dir / "dataset-registry.schema.json"
    ledger_path = output_dir / "consumed-test-ledger.schema.json"
    split_path = output_dir / "split-manifest.schema.json"
    authorization_path = output_dir / "sealed-target-authorization.schema.json"
    selector_path = output_dir / "sealed-target-selector-receipt.schema.json"
    _atomic_write_text(
        registry_path,
        json.dumps(DatasetRegistry.model_json_schema(), indent=2, sort_keys=True) + "\n",
    )
    _atomic_write_text(
        ledger_path,
        json.dumps(ConsumedTestLedger.model_json_schema(), indent=2, sort_keys=True) + "\n",
    )
    _atomic_write_text(
        split_path,
        json.dumps(SplitManifest.model_json_schema(), indent=2, sort_keys=True) + "\n",
    )
    _atomic_write_text(
        authorization_path,
        json.dumps(SealedTargetAuthorization.model_json_schema(), indent=2, sort_keys=True) + "\n",
    )
    _atomic_write_text(
        selector_path,
        json.dumps(SignedSealedTargetSelectorReceipt.model_json_schema(), indent=2, sort_keys=True) + "\n",
    )
    _print(
        {
            "registry_schema": str(registry_path),
            "ledger_schema": str(ledger_path),
            "split_manifest_schema": str(split_path),
            "sealed_target_authorization_schema": str(authorization_path),
            "sealed_target_selector_schema": str(selector_path),
        }
    )


if __name__ == "__main__":
    app()
