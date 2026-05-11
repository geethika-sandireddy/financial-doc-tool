import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, session
from werkzeug.utils import secure_filename

from anomaly import detect_anomalies, extract_transactions
from embeddings import EmbeddingServiceError, get_embedding, search_chunks
from pdf_processor import extract_text_from_pdf

load_dotenv()

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")

UPLOAD_FOLDER = Path(app.config["UPLOAD_FOLDER"])
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {".pdf"}
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


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files are supported"}), 400

    filename = secure_filename(file.filename)
    session_id = get_session_id()
    filepath = UPLOAD_FOLDER / f"{session_id}_{filename}"
    file.save(filepath)

    try:
        chunks = extract_text_from_pdf(str(filepath))
        embeddings = [get_embedding(chunk["content"]) for chunk in chunks]
    except EmbeddingServiceError as exc:
        return jsonify({"error": str(exc)}), 502
    except Exception:
        return jsonify({"error": "Could not process the uploaded PDF"}), 500
    finally:
        filepath.unlink(missing_ok=True)

    document_store[session_id] = {
        "filename": filename,
        "chunks": chunks,
        "embeddings": embeddings,
    }
    return jsonify({"message": f"Processed {len(chunks)} chunks from {filename}"})


@app.route("/search", methods=["POST"])
def search():
    data = request.get_json(silent=True) or {}
    query = str(data.get("query", "")).strip()
    if not query:
        return jsonify({"error": "No query provided"}), 400

    document = get_session_document()
    if document is None:
        return jsonify({"error": "No document uploaded yet"}), 400

    try:
        results = search_chunks(query, document["chunks"], document["embeddings"])
    except EmbeddingServiceError as exc:
        return jsonify({"error": str(exc)}), 502

    return jsonify({"results": results})


@app.route("/anomalies", methods=["GET"])
def anomalies():
    document = get_session_document()
    if document is None:
        return jsonify({"error": "No document uploaded yet"}), 400

    transactions = extract_transactions(document["chunks"])
    normal, flagged = detect_anomalies(transactions)
    return jsonify(
        {
            "total_transactions": len(transactions),
            "normal": len(normal),
            "anomalies": flagged,
        }
    )


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode)
