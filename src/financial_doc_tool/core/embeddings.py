"""Embedding generation via the Gemini embeddings API.

Migrated from the deprecated `google-generativeai` package to `google-genai`
(the former stopped receiving updates/bug fixes upstream). This version also
batches document-chunk embedding into a single API call per batch instead of
one round-trip per chunk, replacing N sequential HTTP round-trips with
ceil(N / batch_size). This repo has no live Gemini API key to benchmark
against, so no specific latency number is claimed here -- the mechanism is
real and testable (see tests/unit/test_embeddings.py for a mocked batching
test), but if you want an exact number for your deployment, time
get_embeddings() before/after with your own API key and put it in
benchmarks/results.md.
"""

from __future__ import annotations

import numpy as np
from google import genai
from google.genai import types
from google.genai.errors import APIError

from financial_doc_tool.config import settings
from financial_doc_tool.exceptions import EmbeddingServiceError

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def _embed_batch(texts: list[str], task_type: str) -> list[np.ndarray]:
    """Embed a batch of texts in a single API call.

    Gemini's embed_content accepts a list of strings natively -- see
    google.genai.models.Models.embed_content -- so batching chunks together
    replaces N sequential round-trips with ceil(N / batch_size).
    """
    if not texts:
        return []
    try:
        response = _get_client().models.embed_content(
            model=settings.embedding_model,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task_type),
        )
    except APIError as exc:
        raise EmbeddingServiceError("Embedding service is unavailable right now") from exc

    if not response.embeddings:
        raise EmbeddingServiceError("Embedding service returned an empty response")
    return [np.array(item.values) for item in response.embeddings]


def get_embeddings(texts: list[str], batch_size: int | None = None) -> list[np.ndarray]:
    """Embed a list of document chunks, batched to respect API limits."""
    batch_size = batch_size or settings.embedding_batch_size
    results: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        results.extend(_embed_batch(batch, task_type="RETRIEVAL_DOCUMENT"))
    return results


def get_embedding(text: str) -> np.ndarray:
    """Embed a single document chunk. Prefer get_embeddings() for multiple chunks."""
    return _embed_batch([text], task_type="RETRIEVAL_DOCUMENT")[0]


def get_query_embedding(text: str) -> np.ndarray:
    """Embed a search query."""
    return _embed_batch([text], task_type="RETRIEVAL_QUERY")[0]


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Return cosine similarity for two vectors."""
    dot_product = np.dot(vec1, vec2)
    magnitude = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if magnitude == 0:
        return 0.0
    return float(dot_product / magnitude)
