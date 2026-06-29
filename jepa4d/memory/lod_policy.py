"""Task-aware memory compression interface."""

from __future__ import annotations


class LODPolicy:
    def compress(self, snapshot: object, task_context: object = None) -> object:
        del task_context
        return snapshot
