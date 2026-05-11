from pdf_processor import chunk_text


def test_chunk_text_keeps_short_text_together():
    text = "alpha beta gamma"

    assert chunk_text(text, chunk_size=100) == ["alpha beta gamma"]


def test_chunk_text_splits_large_text_into_multiple_chunks():
    text = "one two three four five six seven eight nine ten"

    chunks = chunk_text(text, chunk_size=14)

    assert len(chunks) > 1
    assert all(len(chunk) <= 14 for chunk in chunks)
