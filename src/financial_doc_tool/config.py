"""Centralized application settings.

All environment-dependent values are read here, once, instead of scattered
os.getenv() calls throughout the codebase. Import `settings` elsewhere.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    flask_secret_key: str = field(
        default_factory=lambda: os.getenv("FLASK_SECRET_KEY", "dev-secret-key")
    )
    flask_debug: bool = field(
        default_factory=lambda: os.getenv("FLASK_DEBUG", "false").lower() == "true"
    )
    upload_folder: Path = field(default_factory=lambda: Path(os.getenv("UPLOAD_FOLDER", "uploads")))
    max_upload_bytes: int = field(
        default_factory=lambda: int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-004")
    )
    embedding_batch_size: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))
    )
    anomaly_contamination: float = field(
        default_factory=lambda: float(os.getenv("ANOMALY_CONTAMINATION", "0.1"))
    )
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "500")))
    search_top_k: int = field(default_factory=lambda: int(os.getenv("SEARCH_TOP_K", "5")))
    max_pdf_pages: int = field(default_factory=lambda: int(os.getenv("MAX_PDF_PAGES", "500")))


settings = Settings()
