"""In-memory vector index backed by FAISS.

Replaces the original brute-force O(n) cosine-similarity scan in
search_chunks() with an IndexFlatIP (exact inner-product search over
L2-normalized vectors, i.e. exact cosine similarity -- no accuracy is
traded away, only the linear-scan Python loop). See benchmarks/results.md
for query-latency numbers at different corpus sizes.

This is still a single-process, in-memory index (matches the app's current
single-session model, documented as a known limitation in the README). A
persistent deployment would swap this for pgvector or a managed FAISS/HNSW
service without changing the VectorStore interface below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import faiss
import numpy as np


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


@dataclass
class VectorStore:
    """A single document's chunks, indexed for cosine-similarity search."""

    chunks: list[dict[str, Any]] = field(default_factory=list)
    _index: faiss.Index | None = field(default=None, repr=False)
    _dim: int = field(default=0, repr=False)

    def build(self, chunks: list[dict[str, Any]], embeddings: list[np.ndarray]) -> None:
        """Build (or rebuild) the index from chunks and their embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")

        self.chunks = chunks
        if not embeddings:
            self._index = None
            self._dim = 0
            return

        matrix = np.vstack(embeddings).astype("float32")
        matrix = _normalize(matrix)
        self._dim = matrix.shape[1]
        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(matrix)

    @property
    def size(self) -> int:
        return len(self.chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        """Return the top_k most similar chunks to query_embedding."""
        if self._index is None or self.size == 0:
            return []

        query = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        query = _normalize(query)
        k = min(top_k, self.size)
        scores, indices = self._index.search(query, k)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            results.append(
                {
                    "content": chunk["content"],
                    "page": chunk["page"],
                    "source": chunk["source"],
                    "score": round(float(score), 4),
                }
            )
        return results
