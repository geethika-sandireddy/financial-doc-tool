from __future__ import annotations

import io

import numpy as np
import pytest
from reportlab.pdfgen import canvas

from financial_doc_tool.api.app import create_app


def _make_pdf_bytes(lines: list[str]) -> bytes:
    """Build a real single-page PDF with actual extractable text, so
    integration tests exercise the real pypdf extraction path end to end."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=(400, 400))
    y = 380
    for line in lines:
        pdf.drawString(20, y, line)
        y -= 20
    pdf.save()
    return buffer.getvalue()


@pytest.fixture
def minimal_pdf_bytes() -> bytes:
    return _make_pdf_bytes(
        [
            "Vendor Payment Report",
            "Paid $102.50 to Vendor A on invoice 1001",
            "Paid $98.75 to Vendor B on invoice 1002",
            "Paid $101.20 to Vendor C on invoice 1003",
            "Paid $99.00 to Vendor D on invoice 1004",
            "Paid $12500.00 to Vendor E on invoice 1005 -- flagged for review",
        ]
    )


@pytest.fixture
def app(monkeypatch):
    flask_app = create_app()
    flask_app.config.update(TESTING=True, SECRET_KEY="test-secret")
    return flask_app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def fake_embedding_client(monkeypatch):
    """Patch get_embeddings/get_query_embedding so integration tests never
    make real network calls to the Gemini API."""
    import financial_doc_tool.api.routes as routes_module

    def fake_get_embeddings(texts, batch_size=None):
        rng = np.random.default_rng(0)
        return [rng.normal(size=8).astype("float32") for _ in texts]

    def fake_get_query_embedding(text):
        # Deliberately close to the first "chunk" embedding's direction so
        # search tests have a deterministic top result.
        rng = np.random.default_rng(0)
        return rng.normal(size=8).astype("float32")

    monkeypatch.setattr(routes_module, "get_embeddings", fake_get_embeddings)
    monkeypatch.setattr(routes_module, "get_query_embedding", fake_get_query_embedding)
    return None
