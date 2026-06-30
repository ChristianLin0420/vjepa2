"""Ledger-aware dataset access and one-shot sealed-target authorization."""

from __future__ import annotations

import base64
import binascii
import hashlib
import os
import re
from pathlib import Path
from typing import Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from pydantic import Field, field_validator

from jepa4d.validation._content import (
    ContentAddress,
    canonical_json,
    verify_content_addressed_json,
    write_content_addressed_json,
)
from jepa4d.validation.ledger import (
    ConsumedTestLedger,
    FutureUse,
    LedgerState,
    effective_targets,
    load_events,
)
from jepa4d.validation.registry import (
    SHA256_PATTERN,
    TARGET_SPLITS,
    AccessDenied,
    AccessOperation,
    DatasetRegistry,
    StrictModel,
    TargetState,
)

SEALED_AUTHORIZATION_PREFIX = "sealed-target-authorization"
SEALED_AUTHORIZATION_SCHEMA = "jepa4d-sealed-target-authorization-v1"
SEALED_SELECTOR_PREFIX = "sealed-target-selector"
SEALED_SELECTOR_SCHEMA = "jepa4d-signed-sealed-target-selector-receipt-v1"
ONE_SHOT_INTENT = "one-shot-external-evaluation"


class ArtifactBinding(StrictModel):
    """A source artifact identity captured without persisting its local path."""

    name: str
    sha256: str
    bytes: int = Field(ge=1)

    @field_validator("sha256")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("artifact binding requires a lowercase SHA-256 digest")
        return value

    @classmethod
    def from_file(cls, path: str | Path) -> ArtifactBinding:
        source = Path(path).resolve(strict=True)
        if not source.is_file():
            raise ValueError(f"authorization source artifact must be a non-empty file: {source}")
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            before = os.fstat(handle.fileno())
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            after = os.fstat(handle.fileno())
        path_after = source.stat()
        identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in identity_fields) or any(
            getattr(after, field) != getattr(path_after, field) for field in identity_fields
        ):
            raise ValueError(f"authorization source artifact changed while hashing: {source}")
        if after.st_size < 1:
            raise ValueError(f"authorization source artifact must be a non-empty file: {source}")
        return cls(name=source.name, sha256=digest.hexdigest(), bytes=after.st_size)


class SealedTargetSelectorReceipt(StrictModel):
    """Standard selector decision that binds its survivor to every frozen input."""

    schema_version: Literal["jepa4d-sealed-target-selector-receipt-v1"]
    final_authorized: Literal[True]
    dataset_id: str
    split_id: str
    registry_sha256: str
    base_ledger_sha256: str
    event_store_sha256: str
    resolved_instance_sha256: str
    survivor: str
    git_commit: str
    clean_git_commit: Literal[True]
    intent: Literal["one-shot-external-evaluation"]
    preregistration_sha256: str
    checkpoint_sha256: str
    config_sha256: str
    calibrator_sha256: str

    @field_validator(
        "registry_sha256",
        "base_ledger_sha256",
        "event_store_sha256",
        "resolved_instance_sha256",
        "preregistration_sha256",
        "checkpoint_sha256",
        "config_sha256",
        "calibrator_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str) -> str:
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("selector identities must be lowercase SHA-256 digests")
        return value

    @field_validator("git_commit")
    @classmethod
    def validate_git_commit(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError("selector git_commit must be a full 40-character lowercase commit")
        return value


class SignedSealedTargetSelectorReceipt(StrictModel):
    """Ed25519-signed selector authority envelope."""

    schema_version: Literal["jepa4d-signed-sealed-target-selector-receipt-v1"]
    key_id: str
    payload: SealedTargetSelectorReceipt
    signature_ed25519_base64: str

    @field_validator("signature_ed25519_base64")
    @classmethod
    def validate_signature_encoding(cls, value: str) -> str:
        try:
            signature = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError("selector signature must be canonical base64") from error
        if base64.b64encode(signature).decode("ascii") != value:
            raise ValueError("selector signature must be canonical base64")
        if len(signature) != 64:
            raise ValueError("Ed25519 signature must decode to exactly 64 bytes")
        return value


class SealedTargetAuthorization(StrictModel):
    """Typed, content-addressed authority to open one sealed external target once."""

    schema_version: Literal["jepa4d-sealed-target-authorization-v1"]
    dataset_id: str
    split_id: str
    registry_sha256: str
    base_ledger_sha256: str
    event_store_sha256: str
    resolved_instance_sha256: str
    survivor: str
    git_commit: str
    clean_git_commit: Literal[True]
    operation: Literal["external-evaluation"]
    intent: Literal["one-shot-external-evaluation"]
    use_limit: Literal[1]
    preregistration: ArtifactBinding
    selector: ArtifactBinding
    checkpoint: ArtifactBinding
    config: ArtifactBinding
    calibrator: ArtifactBinding
    signed_selector: SignedSealedTargetSelectorReceipt

    @field_validator(
        "registry_sha256",
        "base_ledger_sha256",
        "event_store_sha256",
        "resolved_instance_sha256",
    )
    @classmethod
    def validate_registry_hash(cls, value: str) -> str:
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("authorization registry_sha256 must be a lowercase SHA-256 digest")
        return value

    @field_validator("git_commit")
    @classmethod
    def validate_commit(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError("authorization git_commit must be a full 40-character lowercase commit")
        return value


class GovernedAccessDecision(StrictModel):
    dataset_id: str
    split_id: str
    operation: AccessOperation
    authorized: Literal[True]
    grants_data_access: bool
    registry_sha256: str
    ledger_sha256: str
    ledger_state: LedgerState | None
    future_use: FutureUse | None = None
    sealed_authorization_sha256: str | None = None


def _approved_public_key(
    registry: DatasetRegistry,
    dataset_id: str,
) -> tuple[str, Ed25519PublicKey]:
    entry = registry.dataset(dataset_id)
    authority = entry.sealed_authority
    if authority is None or authority.status != "approved":
        blocker = None if authority is None else authority.blocker
        raise AccessDenied(f"sealed selector authority is not approved for {dataset_id}: {blocker}")
    if authority.key_id is None or authority.public_key_ed25519_base64 is None:
        raise AccessDenied(f"approved sealed authority is incomplete for {dataset_id}")
    public_bytes = base64.b64decode(authority.public_key_ed25519_base64, validate=True)
    return authority.key_id, Ed25519PublicKey.from_public_bytes(public_bytes)


def load_ed25519_private_key_from_env(env_name: str) -> Ed25519PrivateKey:
    """Load a raw 32-byte Ed25519 private key from base64 without persisting or logging it."""
    encoded = os.environ.get(env_name)
    if not encoded:
        raise ValueError(f"Ed25519 private-key environment variable is unset: {env_name}")
    try:
        private_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError(f"{env_name} must contain canonical base64") from error
    if base64.b64encode(private_bytes).decode("ascii") != encoded:
        raise ValueError(f"{env_name} must contain canonical base64")
    if len(private_bytes) != 32:
        raise ValueError(f"{env_name} must decode to exactly 32 private-key bytes")
    return Ed25519PrivateKey.from_private_bytes(private_bytes)


def _verify_signed_selector(
    receipt: SignedSealedTargetSelectorReceipt,
    *,
    registry: DatasetRegistry,
    ledger: ConsumedTestLedger,
    dataset_id: str,
    split_id: str,
) -> SealedTargetSelectorReceipt:
    expected_key_id, public_key = _approved_public_key(registry, dataset_id)
    if receipt.key_id != expected_key_id:
        raise AccessDenied(
            f"selector key-id mismatch for {dataset_id}: expected {expected_key_id}, got {receipt.key_id}"
        )
    payload = receipt.payload
    if payload.dataset_id != dataset_id or payload.split_id != split_id:
        raise AccessDenied(
            "signed selector target mismatch: "
            f"expected {dataset_id}/{split_id}, got {payload.dataset_id}/{payload.split_id}"
        )
    if payload.registry_sha256 != registry.sha256:
        raise AccessDenied("signed selector is bound to a different registry snapshot")
    if payload.event_store_sha256 != ledger.event_store.sha256:
        raise AccessDenied("signed selector is bound to a different canonical event store")
    if payload.resolved_instance_sha256 != ledger.event_store.resolved_instance_sha256:
        raise AccessDenied("signed selector is bound to a different resolved event-store instance")
    if payload.base_ledger_sha256 != ledger.sha256:
        raise AccessDenied("signed selector is bound to a different base ledger")
    try:
        signature = base64.b64decode(receipt.signature_ed25519_base64, validate=True)
        public_key.verify(signature, canonical_json(payload))
    except (InvalidSignature, binascii.Error, ValueError) as error:
        raise AccessDenied("signed selector Ed25519 verification failed") from error
    return payload


def write_sealed_target_selector_receipt(
    *,
    registry: DatasetRegistry,
    ledger: ConsumedTestLedger,
    dataset_id: str,
    split_id: str,
    survivor: str,
    git_commit: str,
    preregistration: str | Path,
    checkpoint: str | Path,
    config: str | Path,
    calibrator: str | Path,
    private_key: Ed25519PrivateKey,
    output_dir: str | Path,
) -> ContentAddress:
    """Write the selector's final one-shot decision as an immutable typed receipt."""
    entry, split = registry.split(dataset_id, split_id)
    ledger.validate_against(registry)
    if split.target_state is not TargetState.SEALED:
        raise ValueError(f"selector receipt requires a sealed split: {dataset_id}/{split_id}")
    if AccessOperation.EXTERNAL_EVALUATION not in split.allowed_operations:
        raise ValueError(f"sealed split does not register external evaluation: {dataset_id}/{split_id}")
    if entry.readiness_blockers:
        raise ValueError(f"dataset audit is incomplete: {list(entry.readiness_blockers)}")
    if not ledger.event_store.externally_append_only:
        raise ValueError("sealed selector issuance requires an approved externally append-only event store")
    if not survivor.strip():
        raise ValueError("survivor cannot be empty")
    preregistration_binding = ArtifactBinding.from_file(preregistration)
    checkpoint_binding = ArtifactBinding.from_file(checkpoint)
    config_binding = ArtifactBinding.from_file(config)
    calibrator_binding = ArtifactBinding.from_file(calibrator)
    if calibrator_binding.sha256 in {
        preregistration_binding.sha256,
        checkpoint_binding.sha256,
        config_binding.sha256,
    }:
        raise ValueError("calibrator must be a distinct frozen artifact")
    key_id, expected_public_key = _approved_public_key(registry, dataset_id)
    actual_public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    expected_public_bytes = expected_public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    if actual_public_bytes != expected_public_bytes:
        raise ValueError("private key does not match the registry-approved Ed25519 public key")
    payload = SealedTargetSelectorReceipt(
        schema_version="jepa4d-sealed-target-selector-receipt-v1",
        final_authorized=True,
        dataset_id=dataset_id,
        split_id=split_id,
        registry_sha256=registry.sha256,
        base_ledger_sha256=ledger.sha256,
        event_store_sha256=ledger.event_store.sha256,
        resolved_instance_sha256=ledger.event_store.resolved_instance_sha256,
        survivor=survivor,
        git_commit=git_commit,
        clean_git_commit=True,
        intent="one-shot-external-evaluation",
        preregistration_sha256=preregistration_binding.sha256,
        checkpoint_sha256=checkpoint_binding.sha256,
        config_sha256=config_binding.sha256,
        calibrator_sha256=calibrator_binding.sha256,
    )
    signed = SignedSealedTargetSelectorReceipt(
        schema_version="jepa4d-signed-sealed-target-selector-receipt-v1",
        key_id=key_id,
        payload=payload,
        signature_ed25519_base64=base64.b64encode(private_key.sign(canonical_json(payload))).decode("ascii"),
    )
    return write_content_addressed_json(signed, output_dir, prefix=SEALED_SELECTOR_PREFIX)


def verify_sealed_target_selector_receipt(
    path: str | Path,
    *,
    registry: DatasetRegistry,
    ledger: ConsumedTestLedger,
    dataset_id: str,
    split_id: str,
) -> tuple[SignedSealedTargetSelectorReceipt, ContentAddress]:
    try:
        source = Path(path).resolve(strict=True)
    except OSError as error:
        raise AccessDenied(f"sealed selector receipt is unavailable: {path}") from error
    value = verify_content_addressed_json(source, prefix=SEALED_SELECTOR_PREFIX)
    receipt = SignedSealedTargetSelectorReceipt.model_validate(value)
    _verify_signed_selector(
        receipt,
        registry=registry,
        ledger=ledger,
        dataset_id=dataset_id,
        split_id=split_id,
    )
    digest = source.stem.removeprefix(f"{SEALED_SELECTOR_PREFIX}-")
    return receipt, ContentAddress(source, digest, source.stat().st_size)


def write_sealed_target_authorization(
    *,
    registry: DatasetRegistry,
    ledger: ConsumedTestLedger,
    dataset_id: str,
    split_id: str,
    survivor: str,
    preregistration: str | Path,
    selector: str | Path,
    checkpoint: str | Path,
    config: str | Path,
    calibrator: str | Path,
    output_dir: str | Path,
) -> ContentAddress:
    """Bind the exact decision/model artifacts and issue a deterministic one-shot authority."""
    entry, split = registry.split(dataset_id, split_id)
    ledger.validate_against(registry)
    if split.target_state is not TargetState.SEALED:
        raise ValueError(f"one-shot authorization requires a sealed split: {dataset_id}/{split_id}")
    if AccessOperation.EXTERNAL_EVALUATION not in split.allowed_operations:
        raise ValueError(f"sealed split does not register external evaluation: {dataset_id}/{split_id}")
    if entry.readiness_blockers:
        raise ValueError(f"dataset audit is incomplete: {list(entry.readiness_blockers)}")
    if not ledger.event_store.externally_append_only:
        raise ValueError("sealed authorization issuance requires an approved externally append-only event store")
    if not survivor.strip():
        raise ValueError("survivor cannot be empty")
    preregistration_binding = ArtifactBinding.from_file(preregistration)
    checkpoint_binding = ArtifactBinding.from_file(checkpoint)
    config_binding = ArtifactBinding.from_file(config)
    calibrator_binding = ArtifactBinding.from_file(calibrator)
    selector_receipt, selector_identity = verify_sealed_target_selector_receipt(
        selector,
        registry=registry,
        ledger=ledger,
        dataset_id=dataset_id,
        split_id=split_id,
    )
    selector_payload = selector_receipt.payload
    expected_selector = {
        "dataset_id": dataset_id,
        "split_id": split_id,
        "registry_sha256": registry.sha256,
        "base_ledger_sha256": ledger.sha256,
        "event_store_sha256": ledger.event_store.sha256,
        "resolved_instance_sha256": ledger.event_store.resolved_instance_sha256,
        "survivor": survivor,
        "git_commit": selector_payload.git_commit,
        "clean_git_commit": True,
        "preregistration_sha256": preregistration_binding.sha256,
        "checkpoint_sha256": checkpoint_binding.sha256,
        "config_sha256": config_binding.sha256,
        "calibrator_sha256": calibrator_binding.sha256,
    }
    observed_selector = {name: getattr(selector_payload, name) for name in expected_selector}
    if observed_selector != expected_selector:
        differences = sorted(
            name for name, expected in expected_selector.items() if observed_selector[name] != expected
        )
        raise ValueError(f"selector receipt does not authorize the supplied survivor/artifacts: {differences}")
    authorization = SealedTargetAuthorization(
        schema_version="jepa4d-sealed-target-authorization-v1",
        dataset_id=dataset_id,
        split_id=split_id,
        registry_sha256=registry.sha256,
        base_ledger_sha256=ledger.sha256,
        event_store_sha256=ledger.event_store.sha256,
        resolved_instance_sha256=ledger.event_store.resolved_instance_sha256,
        survivor=survivor,
        git_commit=selector_payload.git_commit,
        clean_git_commit=True,
        operation="external-evaluation",
        intent="one-shot-external-evaluation",
        use_limit=1,
        preregistration=preregistration_binding,
        selector=ArtifactBinding.from_file(selector_identity.path),
        checkpoint=checkpoint_binding,
        config=config_binding,
        calibrator=calibrator_binding,
        signed_selector=selector_receipt,
    )
    return write_content_addressed_json(authorization, output_dir, prefix=SEALED_AUTHORIZATION_PREFIX)


def verify_sealed_target_authorization(
    path: str | Path,
    *,
    registry: DatasetRegistry,
    ledger: ConsumedTestLedger,
    dataset_id: str,
    split_id: str,
) -> tuple[SealedTargetAuthorization, ContentAddress]:
    """Verify content address, schema, current registry binding, and target identity."""
    try:
        source = Path(path).resolve(strict=True)
    except OSError as error:
        raise AccessDenied(f"sealed authorization artifact is unavailable: {path}") from error
    value = verify_content_addressed_json(source, prefix=SEALED_AUTHORIZATION_PREFIX)
    authorization = SealedTargetAuthorization.model_validate(value)
    if authorization.dataset_id != dataset_id or authorization.split_id != split_id:
        raise AccessDenied(
            "sealed authorization target mismatch: "
            f"expected {dataset_id}/{split_id}, got {authorization.dataset_id}/{authorization.split_id}"
        )
    if authorization.registry_sha256 != registry.sha256:
        raise AccessDenied("sealed authorization is bound to a different registry snapshot")
    if authorization.event_store_sha256 != ledger.event_store.sha256:
        raise AccessDenied("sealed authorization is bound to a different canonical event store")
    if authorization.resolved_instance_sha256 != ledger.event_store.resolved_instance_sha256:
        raise AccessDenied("sealed authorization is bound to a different resolved event-store instance")
    if authorization.base_ledger_sha256 != ledger.sha256:
        raise AccessDenied("sealed authorization is bound to a different base ledger")
    _, split = registry.split(dataset_id, split_id)
    if split.target_state is not TargetState.SEALED:
        raise AccessDenied(f"authorization target is not currently sealed: {dataset_id}/{split_id}")
    selector_payload = _verify_signed_selector(
        authorization.signed_selector,
        registry=registry,
        ledger=ledger,
        dataset_id=dataset_id,
        split_id=split_id,
    )
    expected_bindings = {
        "survivor": selector_payload.survivor,
        "git_commit": selector_payload.git_commit,
        "clean_git_commit": selector_payload.clean_git_commit,
        "base_ledger_sha256": selector_payload.base_ledger_sha256,
        "event_store_sha256": selector_payload.event_store_sha256,
        "resolved_instance_sha256": selector_payload.resolved_instance_sha256,
        "preregistration": selector_payload.preregistration_sha256,
        "checkpoint": selector_payload.checkpoint_sha256,
        "config": selector_payload.config_sha256,
        "calibrator": selector_payload.calibrator_sha256,
    }
    observed_bindings = {
        "survivor": authorization.survivor,
        "git_commit": authorization.git_commit,
        "clean_git_commit": authorization.clean_git_commit,
        "base_ledger_sha256": authorization.base_ledger_sha256,
        "event_store_sha256": authorization.event_store_sha256,
        "resolved_instance_sha256": authorization.resolved_instance_sha256,
        "preregistration": authorization.preregistration.sha256,
        "checkpoint": authorization.checkpoint.sha256,
        "config": authorization.config.sha256,
        "calibrator": authorization.calibrator.sha256,
    }
    if observed_bindings != expected_bindings:
        differences = sorted(
            name for name, expected in expected_bindings.items() if observed_bindings[name] != expected
        )
        raise AccessDenied(f"authorization differs from its signed selector: {differences}")
    selector_payload_bytes = canonical_json(authorization.signed_selector) + b"\n"
    selector_semantic_sha256 = hashlib.sha256(canonical_json(authorization.signed_selector)).hexdigest()
    selector_file_sha256 = hashlib.sha256(selector_payload_bytes).hexdigest()
    if (
        authorization.selector.name != f"{SEALED_SELECTOR_PREFIX}-{selector_semantic_sha256}.json"
        or authorization.selector.sha256 != selector_file_sha256
        or authorization.selector.bytes != len(selector_payload_bytes)
    ):
        raise AccessDenied("authorization selector artifact binding does not match its embedded signed receipt")
    digest = source.stem.removeprefix(f"{SEALED_AUTHORIZATION_PREFIX}-")
    return authorization, ContentAddress(source, digest, source.stat().st_size)


_CONSUMED_FUTURE_OPERATION = {
    AccessOperation.REGRESSION: FutureUse.REGRESSION,
    AccessOperation.MECHANISM_DIAGNOSTIC: FutureUse.MECHANISM_DIAGNOSTIC,
    AccessOperation.REPORTING: FutureUse.REPORTING,
}


class DatasetAccessController:
    """Single public authorization path combining registry, ledger, and event state."""

    def __init__(
        self,
        *,
        registry: DatasetRegistry,
        ledger: ConsumedTestLedger,
    ) -> None:
        ledger.validate_against(registry)
        self.registry = registry
        self.ledger = ledger
        self.events = load_events(registry=registry, ledger=ledger)
        self.targets = effective_targets(registry, ledger, self.events)

    def authorize(
        self,
        dataset_id: str,
        split_id: str,
        operation: AccessOperation | str,
        *,
        sealed_authorization: str | Path | None = None,
    ) -> GovernedAccessDecision:
        operation = AccessOperation(operation)
        key = (dataset_id, split_id)
        _, split = self.registry.split(dataset_id, split_id)
        if key not in self.targets:
            if split.purpose in TARGET_SPLITS:
                raise AccessDenied(f"held-out target is absent from the consumed-test ledger: {dataset_id}/{split_id}")
            if sealed_authorization is not None:
                raise AccessDenied("sealed authorization is invalid for a non-held-out split")
            decision = self.registry._authorize_registered_operation(dataset_id, split_id, operation)
            return GovernedAccessDecision(
                dataset_id=dataset_id,
                split_id=split_id,
                operation=operation,
                authorized=True,
                grants_data_access=decision.grants_data_access,
                registry_sha256=self.registry.sha256,
                ledger_sha256=self.ledger.sha256,
                ledger_state=None,
            )
        target = self.targets[key]

        if target.state is LedgerState.CONSUMED:
            if sealed_authorization is not None:
                raise AccessDenied("sealed authorization cannot be reused after target consumption")
            future_use = _CONSUMED_FUTURE_OPERATION.get(operation)
            if future_use is None:
                raise AccessDenied(
                    f"consumed target denies {operation.value}; only explicitly permitted "
                    "regression, mechanism-diagnostic, or reporting use is possible"
                )
            if future_use not in target.permitted_future_uses:
                raise AccessDenied(
                    f"future use {future_use.value!r} is not permitted for {dataset_id}/{split_id}; "
                    f"allowed={sorted(value.value for value in target.permitted_future_uses)}"
                )
            self.registry._authorize_registered_operation(
                dataset_id,
                split_id,
                operation,
                consumed_future_use=True,
            )
            return GovernedAccessDecision(
                dataset_id=dataset_id,
                split_id=split_id,
                operation=operation,
                authorized=True,
                grants_data_access=operation is not AccessOperation.REPORTING,
                registry_sha256=self.registry.sha256,
                ledger_sha256=self.ledger.sha256,
                ledger_state=target.state,
                future_use=future_use,
            )

        if split.target_state is TargetState.SEALED and operation is AccessOperation.EXTERNAL_EVALUATION:
            raise AccessDenied("sealed external access is granted only by the atomic first-open/consume transition")
        if sealed_authorization is not None:
            raise AccessDenied("sealed authorization may be supplied only for sealed external evaluation")

        decision = self.registry._authorize_registered_operation(
            dataset_id,
            split_id,
            operation,
        )
        return GovernedAccessDecision(
            dataset_id=dataset_id,
            split_id=split_id,
            operation=operation,
            authorized=True,
            grants_data_access=decision.grants_data_access,
            registry_sha256=self.registry.sha256,
            ledger_sha256=self.ledger.sha256,
            ledger_state=target.state,
        )
