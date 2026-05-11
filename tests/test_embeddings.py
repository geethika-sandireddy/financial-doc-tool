import numpy as np

from embeddings import cosine_similarity


def test_cosine_similarity_returns_zero_for_zero_vector():
    left = np.array([0.0, 0.0, 0.0])
    right = np.array([1.0, 2.0, 3.0])

    assert cosine_similarity(left, right) == 0.0


def test_cosine_similarity_returns_one_for_identical_vectors():
    vector = np.array([1.0, 2.0, 3.0])

    assert cosine_similarity(vector, vector) == 1.0
