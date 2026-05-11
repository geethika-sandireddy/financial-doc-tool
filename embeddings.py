import os
from typing import Any

import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


class EmbeddingServiceError(Exception):
    """Raised when the embeddings provider cannot complete a request."""


def _embed_text(text: str, task_type: str) -> np.ndarray:
    """Create an embedding for the given text and Gemini task type."""
    try:
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type=task_type,
        )
    except Exception as exc:
        raise EmbeddingServiceError("Embedding service is unavailable right now") from exc

    embedding = result.get("embedding")
    if embedding is None:
        raise EmbeddingServiceError("Embedding service returned an empty response")
    return np.array(embedding)


def get_embedding(text: str) -> np.ndarray:
    """Create an embedding for a document chunk."""
    return _embed_text(text, "retrieval_document")


def get_query_embedding(text: str) -> np.ndarray:
    """Create an embedding for a search query."""
    return _embed_text(text, "retrieval_query")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Return cosine similarity for two vectors."""
    dot_product = np.dot(vec1, vec2)
    magnitude = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if magnitude == 0:
        return 0.0
    return float(dot_product / magnitude)


def search_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    embeddings: list[np.ndarray],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Search document chunks using cosine similarity over embeddings."""
    query_embedding = get_query_embedding(query)
    scores: list[tuple[int, float]] = []

    for index, embedding in enumerate(embeddings):
        score = cosine_similarity(query_embedding, embedding)
        scores.append((index, score))

    scores.sort(key=lambda item: item[1], reverse=True)

    results: list[dict[str, Any]] = []
    for index, score in scores[:top_k]:
        results.append(
            {
                "content": chunks[index]["content"],
                "page": chunks[index]["page"],
                "source": chunks[index]["source"],
                "score": round(score, 4),
            }
        )

    return results
