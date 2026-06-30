"""Dataset governance primitives for the staged JEPA-4D validation program."""

from jepa4d.validation.access import DatasetAccessController, SealedTargetAuthorization
from jepa4d.validation.ledger import ConsumedTestLedger
from jepa4d.validation.registry import DatasetRegistry

__all__ = [
    "ConsumedTestLedger",
    "DatasetAccessController",
    "DatasetRegistry",
    "SealedTargetAuthorization",
]
