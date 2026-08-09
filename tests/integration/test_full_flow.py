"""Integration tests exercising the real Flask routes end to end: a client
uploads a PDF, searches it, and requests the anomaly report, over real HTTP
requests against the app's test client. The Gemini embedding calls are
mocked (see conftest.py) so these run offline and deterministically; the PDF
parsing, chunking, FAISS indexing, and anomaly detection are all real.
"""

from __future__ import annotations

import io


def test_upload_then_search_then_anomalies_full_flow(
    client, fake_embedding_client, minimal_pdf_bytes
):
    upload_response = client.post(
        "/upload",
        data={"file": (io.BytesIO(minimal_pdf_bytes), "report.pdf")},
        content_type="multipart/form-data",
    )
    assert upload_response.status_code == 200
    assert "Processed" in upload_response.get_json()["message"]

    search_response = client.post("/search", json={"query": "vendor payment"})
    assert search_response.status_code == 200
    results = search_response.get_json()["results"]
    assert len(results) > 0
    assert all("score" in r and "content" in r for r in results)

    anomalies_response = client.get("/anomalies")
    assert anomalies_response.status_code == 200
    payload = anomalies_response.get_json()
    assert payload["total_transactions"] >= 5
    # The fixture PDF has one deliberately huge value (12500.00) among
    # four ~100-value entries -- exactly the shape the anomaly eval harness
    # in tests/performance/ validates the detector catches.
    assert payload["total_transactions"] == payload["normal"] + len(payload["anomalies"])


def test_search_without_upload_returns_400(client, fake_embedding_client):
    response = client.post("/search", json={"query": "anything"})
    assert response.status_code == 400
    assert "No document" in response.get_json()["error"]


def test_anomalies_without_upload_returns_400(client, fake_embedding_client):
    response = client.get("/anomalies")
    assert response.status_code == 400


def test_upload_rejects_non_pdf(client, fake_embedding_client):
    response = client.post(
        "/upload",
        data={"file": (io.BytesIO(b"not a pdf"), "notes.txt")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "Only PDF" in response.get_json()["error"]


def test_upload_rejects_corrupted_pdf(client, fake_embedding_client):
    response = client.post(
        "/upload",
        data={"file": (io.BytesIO(b"%PDF-1.4 not actually a real pdf body"), "broken.pdf")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 422
