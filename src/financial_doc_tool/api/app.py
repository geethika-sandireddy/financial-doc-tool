"""Flask application factory.

Kept separate from run.py so tests can import create_app() directly without
starting a dev server (see tests/integration/conftest.py).
"""

from __future__ import annotations

from pathlib import Path

from flask import Flask

from financial_doc_tool.api.routes import bp
from financial_doc_tool.config import settings

_TEMPLATE_FOLDER = Path(__file__).resolve().parents[3] / "templates"


def create_app() -> Flask:
    app = Flask(__name__, template_folder=str(_TEMPLATE_FOLDER))
    app.config["MAX_CONTENT_LENGTH"] = settings.max_upload_bytes
    app.secret_key = settings.flask_secret_key
    app.register_blueprint(bp)
    return app
