"""Benchmarks search latency: the original brute-force Python cosine loop
vs. the FAISS-backed VectorStore, at increasing corpus sizes. Asserts FAISS
stays meaningfully faster as a regression guard, and prints a full table
when run directly.

Run directly for a human-readable report:
    python tests/performance/test_search_latency.py
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from financial_doc_tool.core.vector_store import VectorStore


def _brute_force_search(query: np.ndarray, embeddings: list[np.ndarray], top_k: int = 5):
    def cos(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom else 0.0

    scores = [(i, cos(query, e)) for i, e in enumerate(embeddings)]
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores[:top_k]


def _timed_run(fn, repeats: int = 5) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats


def _build_corpus(n_chunks: int, dim: int = 768, seed: int = 42):
    rng = np.random.default_rng(seed)
    embeddings = [rng.normal(size=dim).astype("float32") for _ in range(n_chunks)]
    chunks = [{"content": f"chunk {i}", "page": 1, "source": "doc.pdf"} for i in range(n_chunks)]
    query = rng.normal(size=dim).astype("float32")
    return chunks, embeddings, query


@pytest.mark.parametrize("n_chunks", [1000, 10000])
def test_faiss_search_is_faster_than_brute_force(n_chunks: int) -> None:
    chunks, embeddings, query = _build_corpus(n_chunks)

    brute_force_time = _timed_run(lambda: _brute_force_search(query, embeddings, top_k=5))

    store = VectorStore()
    store.build(chunks, embeddings)
    faiss_time = _timed_run(lambda: store.search(query, top_k=5))

    # Regression guard: FAISS should stay at least 5x faster at this scale.
    # Measured speedups on the reference machine were 25-30x; 5x leaves
    # comfortable headroom for slower CI runners.
    assert faiss_time < brute_force_time / 5


def _time_brute_force_for_corpus(query: np.ndarray, embeddings: list[np.ndarray]) -> float:
    return _timed_run(lambda: _brute_force_search(query, embeddings, top_k=5))


def _time_faiss_for_corpus(store: VectorStore, query: np.ndarray) -> float:
    return _timed_run(lambda: store.search(query, top_k=5))


if __name__ == "__main__":
    print(f"{'chunks':>8} | {'brute-force (ms)':>17} | {'FAISS (ms)':>11} | {'speedup':>8}")
    print("-" * 56)
    for n in (100, 1000, 10000, 50000):
        chunks, embeddings, query = _build_corpus(n)
        bf_time = _time_brute_force_for_corpus(query, embeddings)
        store = VectorStore()
        store.build(chunks, embeddings)
        faiss_time = _time_faiss_for_corpus(store, query)
        print(
            f"{n:>8} | {bf_time * 1000:>17.3f} | {faiss_time * 1000:>11.3f} | "
            f"{bf_time / faiss_time:>7.1f}x"
        )
