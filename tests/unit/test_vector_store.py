import numpy as np

from financial_doc_tool.core.vector_store import VectorStore


def _chunk(content: str, page: int = 1, source: str = "doc.pdf") -> dict:
    return {"content": content, "page": page, "source": source}


def test_empty_store_returns_no_results():
    store = VectorStore()
    store.build([], [])

    results = store.search(np.array([1.0, 0.0, 0.0]), top_k=5)

    assert results == []


def test_rejects_mismatched_chunks_and_embeddings():
    store = VectorStore()
    try:
        store.build([_chunk("a")], [])
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_search_returns_most_similar_chunk_first():
    chunks = [_chunk("alpha"), _chunk("beta"), _chunk("gamma")]
    embeddings = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.9, 0.1, 0.0]),  # close to the query, should rank highly
    ]
    store = VectorStore()
    store.build(chunks, embeddings)

    results = store.search(np.array([1.0, 0.0, 0.0]), top_k=2)

    assert len(results) == 2
    assert results[0]["content"] == "alpha"
    assert results[0]["score"] > results[1]["score"]


def test_top_k_is_capped_at_store_size():
    chunks = [_chunk("only one chunk")]
    embeddings = [np.array([1.0, 0.0])]
    store = VectorStore()
    store.build(chunks, embeddings)

    results = store.search(np.array([1.0, 0.0]), top_k=10)

    assert len(results) == 1


def test_size_reflects_chunk_count():
    store = VectorStore()
    store.build([_chunk("a"), _chunk("b")], [np.array([1.0, 0.0]), np.array([0.0, 1.0])])

    assert store.size == 2
