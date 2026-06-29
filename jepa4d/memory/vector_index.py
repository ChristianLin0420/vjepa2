"""Dependency-free cosine index with a future FAISS-compatible boundary."""

from __future__ import annotations

import numpy as np


class VectorIndex:
    def __init__(self) -> None:
        self.vectors: dict[str, np.ndarray] = {}

    def add(self, key: str, vector: np.ndarray) -> None:
        self.vectors[key] = np.asarray(vector, dtype=np.float32)

    def search(self, query: np.ndarray, limit: int = 10) -> list[tuple[str, float]]:
        query = np.asarray(query, dtype=np.float32)
        query = query / max(float(np.linalg.norm(query)), 1e-8)
        scored = [
            (key, float(query @ (value / max(float(np.linalg.norm(value)), 1e-8))))
            for key, value in self.vectors.items()
        ]
        return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]
