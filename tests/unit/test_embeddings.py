import numpy as np

from financial_doc_tool.core.embeddings import cosine_similarity


def test_cosine_similarity_returns_zero_for_zero_vector():
    left = np.array([0.0, 0.0, 0.0])
    right = np.array([1.0, 2.0, 3.0])

    assert cosine_similarity(left, right) == 0.0


def test_cosine_similarity_returns_one_for_identical_vectors():
    vector = np.array([1.0, 2.0, 3.0])

    assert cosine_similarity(vector, vector) == 1.0


def test_get_embeddings_batches_instead_of_one_call_per_chunk(monkeypatch):
    """Regression guard for the batching fix: 250 chunks at batch_size=100
    must be 3 API calls (100 + 100 + 50), not 250."""
    import financial_doc_tool.core.embeddings as embeddings_module

    call_sizes: list[int] = []

    class _FakeEmbedding:
        def __init__(self, values):
            self.values = values

    class _FakeResponse:
        def __init__(self, n):
            self.embeddings = [_FakeEmbedding([1.0, 0.0, 0.0]) for _ in range(n)]

    class _FakeModels:
        def embed_content(self, *, model, contents, config):
            call_sizes.append(len(contents))
            return _FakeResponse(len(contents))

    class _FakeClient:
        models = _FakeModels()

    monkeypatch.setattr(embeddings_module, "_get_client", lambda: _FakeClient())

    texts = [f"chunk {i}" for i in range(250)]
    results = embeddings_module.get_embeddings(texts, batch_size=100)

    assert call_sizes == [100, 100, 50]
    assert len(results) == 250
