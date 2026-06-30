"""Append-only consumed-target ledger with first-open lineage."""

from __future__ import annotations

import fcntl
import os
import re
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from jepa4d.validation._content import (
    ContentAddress,
    load_yaml_unique,
    sha256_value,
    verify_content_addressed_json,
    write_content_addressed_json,
)
from jepa4d.validation.registry import (
    ENV_PATTERN,
    SHA256_PATTERN,
    TARGET_SPLITS,
    AccessOperation,
    DatasetRegistry,
    StrictModel,
    TargetState,
)

LEDGER_SCHEMA_VERSION = "jepa4d-consumed-test-ledger-v1"
EVENT_SCHEMA_VERSION = "jepa4d-consumed-test-event-v1"


class EventStoreUnavailable(ValueError):
    """Raised when the deployment has not configured the registry-governed event-store root."""


class LedgerState(StrEnum):
    PLANNED_UNAVAILABLE = "planned-unavailable"
    AVAILABLE_UNOPENED = "available-unopened"
    SEALED_UNOPENED = "sealed-unopened"
    SERVER_ONLY_UNOPENED = "server-only-unopened"
    CONSUMED = "consumed"


class TimePrecision(StrEnum):
    EXACT = "exact"
    DAY = "day"
    UNKNOWN = "unknown"


class FutureUse(StrEnum):
    REGRESSION = "regression"
    MECHANISM_DIAGNOSTIC = "mechanism-diagnostic"
    REPRODUCIBILITY_AUDIT = "reproducibility-audit"
    REPORTING = "reporting"
    NO_FUTURE_USE = "no-future-use"


class EventStoreSpec(StrictModel):
    """Portable identity for the one canonical local event store."""

    root_env: str
    relative_path: str
    durability: Literal["local-filesystem-best-effort", "external-append-only"]
    externally_append_only: bool
    deployment_blocker: str | None = None

    @field_validator("root_env")
    @classmethod
    def validate_root_env(cls, value: str) -> str:
        if not ENV_PATTERN.fullmatch(value):
            raise ValueError("event-store root must be an environment-variable name")
        return value

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or not value or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("event-store relative_path must be a safe non-empty relative path")
        return path.as_posix()

    @model_validator(mode="after")
    def validate_durability(self) -> EventStoreSpec:
        if self.externally_append_only:
            if self.durability != "external-append-only":
                raise ValueError("externally append-only stores require durability='external-append-only'")
            if self.deployment_blocker is not None:
                raise ValueError("an approved externally append-only store cannot retain a deployment blocker")
        else:
            if self.durability != "local-filesystem-best-effort":
                raise ValueError("local rollbackable stores require durability='local-filesystem-best-effort'")
            if not self.deployment_blocker:
                raise ValueError("local rollbackable stores require an explicit deployment blocker")
        return self

    @property
    def sha256(self) -> str:
        return sha256_value(self)

    @property
    def resolved_instance_sha256(self) -> str:
        """Bind one deployment's canonical resolved store without publishing its local path."""
        return sha256_value(
            {
                "event_store_sha256": self.sha256,
                "resolved_path": self.resolve().as_posix(),
            }
        )

    def resolve(self, environ: Mapping[str, str] | None = None) -> Path:
        values = os.environ if environ is None else environ
        root_value = values.get(self.root_env)
        if not root_value:
            raise EventStoreUnavailable(f"canonical event-store root environment variable is unset: {self.root_env}")
        root = Path(root_value).expanduser().resolve()
        target = (root / self.relative_path).resolve()
        if target != root and root not in target.parents:
            raise ValueError("resolved event store escapes its configured root")
        return target


class FirstOpenLineage(StrictModel):
    opened_at: str
    time_precision: TimePrecision
    experiment_id: str
    git_commit: str
    execution_id: str | None = None
    source_record: str
    source_record_sha256: str | None = None
    registry_sha256: str | None = None
    imported_historical: bool = False
    notes: str

    @field_validator("source_record_sha256", "registry_sha256")
    @classmethod
    def validate_optional_digest(cls, value: str | None) -> str | None:
        if value is not None and not SHA256_PATTERN.fullmatch(value):
            raise ValueError("lineage hashes must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def validate_timestamp_and_commit(self) -> FirstOpenLineage:
        if self.time_precision is TimePrecision.DAY:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.opened_at):
                raise ValueError("day-precision opened_at must be YYYY-MM-DD")
        elif self.time_precision is TimePrecision.EXACT:
            if "T" not in self.opened_at or not (self.opened_at.endswith("Z") or "+" in self.opened_at[10:]):
                raise ValueError("exact opened_at must be an ISO-8601 timestamp with timezone")
        elif self.opened_at != "unknown":
            raise ValueError("unknown time precision requires opened_at='unknown'")
        if self.git_commit != "unknown" and not re.fullmatch(r"[0-9a-f]{7,40}", self.git_commit):
            raise ValueError("git_commit must be 7-40 lowercase hex characters or 'unknown'")
        return self


class LedgerTarget(StrictModel):
    dataset_id: str
    split_id: str
    state: LedgerState
    first_open: FirstOpenLineage | None = None
    permitted_future_uses: frozenset[FutureUse] = frozenset()
    seal_evidence: str | None = None
    notes: str

    @model_validator(mode="after")
    def state_is_coherent(self) -> LedgerTarget:
        if self.state is LedgerState.CONSUMED:
            if self.first_open is None or not self.permitted_future_uses:
                raise ValueError("consumed targets require first_open lineage and explicit future-use policy")
            if FutureUse.NO_FUTURE_USE in self.permitted_future_uses and len(self.permitted_future_uses) != 1:
                raise ValueError("no-future-use cannot be combined with another future use")
        elif self.first_open is not None or self.permitted_future_uses:
            raise ValueError("unopened targets cannot contain first-open lineage or future reuse permissions")
        if self.state is LedgerState.SEALED_UNOPENED and not self.seal_evidence:
            raise ValueError("sealed targets require seal_evidence")
        return self


class ConsumedTestLedger(StrictModel):
    schema_version: Literal["jepa4d-consumed-test-ledger-v1"]
    ledger_version: str
    event_store: EventStoreSpec
    targets: tuple[LedgerTarget, ...]

    @model_validator(mode="after")
    def reject_duplicate_targets(self) -> ConsumedTestLedger:
        seen: set[tuple[str, str]] = set()
        for target in self.targets:
            key = (target.dataset_id, target.split_id)
            if key in seen:
                raise ValueError(f"duplicate ledger target: {target.dataset_id}/{target.split_id}")
            seen.add(key)
        return self

    @classmethod
    def load(cls, path: str | Path) -> ConsumedTestLedger:
        return cls.model_validate(load_yaml_unique(path))

    @property
    def sha256(self) -> str:
        return sha256_value(self)

    def target(self, dataset_id: str, split_id: str) -> LedgerTarget:
        for target in self.targets:
            if (target.dataset_id, target.split_id) == (dataset_id, split_id):
                return target
        raise KeyError(f"target is absent from ledger: {dataset_id}/{split_id}")

    def validate_against(self, registry: DatasetRegistry) -> None:
        ledger_keys = {(target.dataset_id, target.split_id) for target in self.targets}
        required_keys = {
            (entry.dataset_id, split.split_id)
            for entry in registry.datasets
            for split in entry.splits
            if split.purpose in TARGET_SPLITS
        }
        missing = sorted(required_keys - ledger_keys)
        if missing:
            raise ValueError(f"ledger is missing held-out target rows: {missing}")
        for target in self.targets:
            _, split = registry.split(target.dataset_id, target.split_id)
            expected = {
                TargetState.SEALED: LedgerState.SEALED_UNOPENED,
                TargetState.SERVER_ONLY: LedgerState.SERVER_ONLY_UNOPENED,
                TargetState.UNAVAILABLE: LedgerState.PLANNED_UNAVAILABLE,
            }.get(split.target_state)
            if target.state is not LedgerState.CONSUMED and expected is not None and target.state is not expected:
                raise ValueError(
                    f"ledger state {target.state.value} conflicts with registry state {split.target_state.value} "
                    f"for {target.dataset_id}/{target.split_id}"
                )


class ConsumptionEvent(StrictModel):
    schema_version: Literal["jepa4d-consumed-test-event-v1"]
    dataset_id: str
    split_id: str
    prior_state: LedgerState
    open_operation: AccessOperation
    first_open: FirstOpenLineage
    permitted_future_uses: frozenset[FutureUse] = Field(min_length=1)
    registry_sha256: str
    base_ledger_sha256: str
    event_store_sha256: str
    resolved_instance_sha256: str
    sealed_authorization_sha256: str | None = None

    @field_validator(
        "registry_sha256",
        "base_ledger_sha256",
        "event_store_sha256",
        "resolved_instance_sha256",
        "sealed_authorization_sha256",
    )
    @classmethod
    def validate_digest(cls, value: str | None) -> str | None:
        if value is not None and not SHA256_PATTERN.fullmatch(value):
            raise ValueError("event identities must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def validate_future_use(self) -> ConsumptionEvent:
        if self.prior_state is LedgerState.CONSUMED:
            raise ValueError("consumption event cannot transition an already consumed target")
        if FutureUse.NO_FUTURE_USE in self.permitted_future_uses and len(self.permitted_future_uses) != 1:
            raise ValueError("no-future-use cannot be combined with another future use")
        is_sealed_external = (
            self.prior_state is LedgerState.SEALED_UNOPENED
            and self.open_operation is AccessOperation.EXTERNAL_EVALUATION
        )
        if is_sealed_external and self.sealed_authorization_sha256 is None:
            raise ValueError("sealed external consumption requires its typed authorization artifact SHA-256")
        if self.sealed_authorization_sha256 is not None and not is_sealed_external:
            raise ValueError("sealed authorization identity is legal only for sealed external consumption")
        if self.first_open.registry_sha256 != self.registry_sha256:
            raise ValueError("first_open.registry_sha256 must match the event registry_sha256")
        return self


def load_events(*, registry: DatasetRegistry, ledger: ConsumedTestLedger) -> tuple[ConsumptionEvent, ...]:
    directory = ledger.event_store.resolve()
    if not directory.exists():
        return ()
    events: list[ConsumptionEvent] = []
    seen: set[tuple[str, str]] = set()
    for path in sorted(directory.glob("consumption-*.json")):
        value = verify_content_addressed_json(path, prefix="consumption")
        event = ConsumptionEvent.model_validate(value)
        if event.registry_sha256 != registry.sha256:
            raise ValueError(
                f"event registry_sha256 mismatch for {event.dataset_id}/{event.split_id}: "
                f"expected {registry.sha256}, got {event.registry_sha256}"
            )
        if event.base_ledger_sha256 != ledger.sha256:
            raise ValueError(
                f"event base_ledger_sha256 mismatch for {event.dataset_id}/{event.split_id}: "
                f"expected {ledger.sha256}, got {event.base_ledger_sha256}"
            )
        if event.event_store_sha256 != ledger.event_store.sha256:
            raise ValueError(
                f"event event_store_sha256 mismatch for {event.dataset_id}/{event.split_id}: "
                f"expected {ledger.event_store.sha256}, got {event.event_store_sha256}"
            )
        if event.resolved_instance_sha256 != ledger.event_store.resolved_instance_sha256:
            raise ValueError(
                f"event resolved_instance_sha256 mismatch for {event.dataset_id}/{event.split_id}: "
                f"expected {ledger.event_store.resolved_instance_sha256}, got {event.resolved_instance_sha256}"
            )
        key = (event.dataset_id, event.split_id)
        if key in seen:
            raise ValueError(f"multiple first-open events for {event.dataset_id}/{event.split_id}")
        seen.add(key)
        events.append(event)
    return tuple(events)


def effective_targets(
    registry: DatasetRegistry,
    ledger: ConsumedTestLedger,
    events: tuple[ConsumptionEvent, ...],
) -> dict[tuple[str, str], LedgerTarget]:
    ledger.validate_against(registry)
    effective = {(value.dataset_id, value.split_id): value for value in ledger.targets}
    for event in events:
        if event.registry_sha256 != registry.sha256:
            raise ValueError(
                f"event registry_sha256 mismatch for {event.dataset_id}/{event.split_id}: "
                f"expected {registry.sha256}, got {event.registry_sha256}"
            )
        if event.base_ledger_sha256 != ledger.sha256:
            raise ValueError(
                f"event base_ledger_sha256 mismatch for {event.dataset_id}/{event.split_id}: "
                f"expected {ledger.sha256}, got {event.base_ledger_sha256}"
            )
        if event.event_store_sha256 != ledger.event_store.sha256:
            raise ValueError(
                f"event event_store_sha256 mismatch for {event.dataset_id}/{event.split_id}: "
                f"expected {ledger.event_store.sha256}, got {event.event_store_sha256}"
            )
        if event.resolved_instance_sha256 != ledger.event_store.resolved_instance_sha256:
            raise ValueError(
                f"event resolved_instance_sha256 mismatch for {event.dataset_id}/{event.split_id}: "
                f"expected {ledger.event_store.resolved_instance_sha256}, got {event.resolved_instance_sha256}"
            )
        key = (event.dataset_id, event.split_id)
        if key not in effective:
            raise ValueError(f"event references target absent from base ledger: {event.dataset_id}/{event.split_id}")
        prior = effective[key]
        if prior.state is LedgerState.CONSUMED:
            raise ValueError(f"target already consumed in base ledger: {event.dataset_id}/{event.split_id}")
        if event.prior_state is not prior.state:
            raise ValueError(f"event prior_state does not match base ledger for {event.dataset_id}/{event.split_id}")
        effective[key] = LedgerTarget(
            dataset_id=event.dataset_id,
            split_id=event.split_id,
            state=LedgerState.CONSUMED,
            first_open=event.first_open,
            permitted_future_uses=event.permitted_future_uses,
            notes="Consumed by immutable first-open event.",
        )
    return effective


def append_first_open(
    *,
    registry: DatasetRegistry,
    ledger: ConsumedTestLedger,
    dataset_id: str,
    split_id: str,
    operation: AccessOperation | str,
    lineage: FirstOpenLineage,
    permitted_future_uses: frozenset[FutureUse],
    sealed_authorization: str | Path | None = None,
) -> ContentAddress:
    """Record exactly one immutable first-open event under an inter-process lock."""
    ledger.validate_against(registry)
    operation = AccessOperation(operation)
    directory = ledger.event_store.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".consumption.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        events = load_events(registry=registry, ledger=ledger)
        effective = effective_targets(registry, ledger, events)
        key = (dataset_id, split_id)
        if key not in effective:
            raise KeyError(f"target is absent from ledger: {dataset_id}/{split_id}")
        prior = effective[key]
        if prior.state is LedgerState.CONSUMED:
            raise ValueError(f"target already consumed: {dataset_id}/{split_id}")
        # Import lazily to avoid a module cycle. Sealed access is deliberately
        # verified only inside this lock-held transition and has no plain authorize path.
        from jepa4d.validation.access import DatasetAccessController, verify_sealed_target_authorization

        _, split = registry.split(dataset_id, split_id)
        sealed_identity: str | None = None
        if split.target_state is TargetState.SEALED and operation is AccessOperation.EXTERNAL_EVALUATION:
            if not ledger.event_store.externally_append_only:
                raise PermissionError(
                    "sealed target first-open requires an approved externally append-only event store"
                )
            if sealed_authorization is None:
                raise PermissionError(
                    "sealed target requires an authenticated content-addressed authorization artifact"
                )
            _, identity = verify_sealed_target_authorization(
                sealed_authorization,
                registry=registry,
                ledger=ledger,
                dataset_id=dataset_id,
                split_id=split_id,
            )
            decision = registry._authorize_registered_operation(
                dataset_id,
                split_id,
                AccessOperation.EXTERNAL_EVALUATION,
                sealed_authorization_verified=True,
            )
            sealed_identity = identity.sha256
            grants_data_access = decision.grants_data_access
        else:
            controller = DatasetAccessController(registry=registry, ledger=ledger)
            governed_decision = controller.authorize(
                dataset_id,
                split_id,
                operation,
                sealed_authorization=sealed_authorization,
            )
            grants_data_access = governed_decision.grants_data_access
        if not grants_data_access:
            raise PermissionError(
                f"operation {operation.value!r} does not open target data and cannot create a consumption event"
            )
        if lineage.registry_sha256 is not None and lineage.registry_sha256 != registry.sha256:
            raise ValueError("first-open lineage names a different registry snapshot")
        event = ConsumptionEvent(
            schema_version="jepa4d-consumed-test-event-v1",
            dataset_id=dataset_id,
            split_id=split_id,
            prior_state=prior.state,
            open_operation=operation,
            first_open=lineage.model_copy(update={"registry_sha256": registry.sha256}),
            permitted_future_uses=permitted_future_uses,
            registry_sha256=registry.sha256,
            base_ledger_sha256=ledger.sha256,
            event_store_sha256=ledger.event_store.sha256,
            resolved_instance_sha256=ledger.event_store.resolved_instance_sha256,
            sealed_authorization_sha256=sealed_identity,
        )
        artifact = write_content_addressed_json(event, directory, prefix="consumption")
        # Verify the complete event set before releasing the lock.
        validated_events = load_events(registry=registry, ledger=ledger)
        effective_targets(registry, ledger, validated_events)
        return artifact


def freeze_ledger(
    registry: DatasetRegistry,
    ledger: ConsumedTestLedger,
    output_dir: str | Path,
) -> ContentAddress:
    events = load_events(registry=registry, ledger=ledger)
    effective = effective_targets(registry, ledger, events)
    payload = {
        "schema_version": "jepa4d-consumed-test-ledger-snapshot-v1",
        "base_ledger_sha256": ledger.sha256,
        "event_store_sha256": ledger.event_store.sha256,
        "resolved_instance_sha256": ledger.event_store.resolved_instance_sha256,
        "durability": ledger.event_store.durability,
        "externally_append_only": ledger.event_store.externally_append_only,
        "deployment_blocker": ledger.event_store.deployment_blocker,
        "event_sha256": sorted(sha256_value(event) for event in events),
        "targets": [
            value.model_dump(mode="json", exclude_none=False)
            for _, value in sorted(effective.items(), key=lambda item: item[0])
        ],
    }
    return write_content_addressed_json(payload, output_dir, prefix="consumed-test-ledger")
