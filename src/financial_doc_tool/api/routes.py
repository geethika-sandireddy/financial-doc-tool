"""HTTP layer only. No business logic lives here -- every route delegates to
financial_doc_tool.core modules and translates results/exceptions into JSON
responses. This split is what makes core/ unit-testable without spinning up
Flask at all (see tests/unit/) and lets tests/integration/ exercise the full
upload -> search -> anomaly flow through real HTTP requests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from flask import Blueprint, Response, jsonify, render_template, request, session
from werkzeug.utils import secure_filename

from financial_doc_tool.config import settings
from financial_doc_tool.core.anomaly import detect_anomalies, extract_transactions
from financial_doc_tool.core.embeddings import get_embeddings, get_query_embedding
from financial_doc_tool.core.pdf_processor import extract_text_from_pdf
from financial_doc_tool.core.vector_store import VectorStore
from financial_doc_tool.exceptions import EmbeddingServiceError, PdfProcessingError

bp = Blueprint("financial_doc_tool", __name__)

ALLOWED_EXTENSIONS = {".pdf"}
# Session state, in-process only -- see README "Limitations" for what this
# means for multi-instance deployments.
document_store: dict[str, dict[str, Any]] = {}


def get_session_id() -> str:
    """Return a stable session id for the current browser session."""
    session_id = session.get("session_id")
    if session_id is None:
        session_id = uuid4().hex
        session["session_id"] = session_id
    return session_id


def get_session_document() -> dict[str, Any] | None:
    """Fetch the uploaded document state for the current session."""
    return document_store.get(get_session_id())


def allowed_file(filename: str) -> bool:
    """Allow uploads only for PDF files."""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@bp.route("/")
def index() -> str:
    return render_template("index.html")


@bp.route("/upload", methods=["POST"])
def upload() -> tuple[Response, int] | Response:
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files are supported"}), 400

    filename = secure_filename(file.filename)
    session_id = get_session_id()
    settings.upload_folder.mkdir(exist_ok=True)
    filepath = settings.upload_folder / f"{session_id}_{filename}"
    file.save(filepath)

    try:
        chunks = extract_text_from_pdf(str(filepath))
        embeddings = get_embeddings([chunk["content"] for chunk in chunks])
    except PdfProcessingError as exc:
        return jsonify({"error": str(exc)}), 422
    except EmbeddingServiceError as exc:
        return jsonify({"error": str(exc)}), 502
    finally:
        filepath.unlink(missing_ok=True)

    store = VectorStore()
    store.build(chunks, embeddings)
    document_store[session_id] = {"filename": filename, "store": store}
    return jsonify({"message": f"Processed {len(chunks)} chunks from {filename}"})


@bp.route("/search", methods=["POST"])
def search() -> tuple[Response, int] | Response:
    data = request.get_json(silent=True) or {}
    query = str(data.get("query", "")).strip()
    if not query:
        return jsonify({"error": "No query provided"}), 400

    document = get_session_document()
    if document is None:
        return jsonify({"error": "No document uploaded yet"}), 400

    try:
        query_embedding = get_query_embedding(query)
    except EmbeddingServiceError as exc:
        return jsonify({"error": str(exc)}), 502

    results = document["store"].search(query_embedding, top_k=settings.search_top_k)
    return jsonify({"results": results})


@bp.route("/anomalies", methods=["GET"])
def anomalies() -> tuple[Response, int] | Response:
    document = get_session_document()
    if document is None:
        return jsonify({"error": "No document uploaded yet"}), 400

    transactions = extract_transactions(document["store"].chunks)
    normal, flagged = detect_anomalies(transactions)
    return jsonify(
        {
            "total_transactions": len(transactions),
            "normal": len(normal),
            "anomalies": flagged,
        }
    )
