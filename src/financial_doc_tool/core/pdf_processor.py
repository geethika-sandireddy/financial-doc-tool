from __future__ import annotations

from pathlib import Path

import pypdf

from financial_doc_tool.config import settings
from financial_doc_tool.exceptions import PdfProcessingError


def extract_text_from_pdf(
    pdf_path: str, max_pages: int | None = None
) -> list[dict[str, str | int]]:
    """Extract text from a PDF and return chunk metadata.

    Raises PdfProcessingError if the PDF is unreadable/corrupted or exceeds
    max_pages, instead of letting a raw pypdf exception (or an unbounded
    parse of an enormous file) propagate up to the Flask layer.
    """
    max_pages = max_pages if max_pages is not None else settings.max_pdf_pages
    text_chunks: list[dict[str, str | int]] = []

    try:
        with open(pdf_path, "rb") as file:
            reader = pypdf.PdfReader(file)

            if len(reader.pages) > max_pages:
                raise PdfProcessingError(
                    f"PDF has {len(reader.pages)} pages, which exceeds the "
                    f"{max_pages}-page limit for this deployment."
                )

            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    for chunk in chunk_text(text, chunk_size=settings.chunk_size):
                        text_chunks.append(
                            {
                                "content": chunk,
                                "page": page_num,
                                "source": Path(pdf_path).name,
                            }
                        )
    except PdfProcessingError:
        raise
    except pypdf.errors.PdfReadError as exc:
        raise PdfProcessingError("The uploaded file is not a valid or readable PDF.") from exc

    return text_chunks


def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    """Split text into chunks that stay close to the target character length."""
    words = text.split()
    chunks: list[str] = []
    current_words: list[str] = []
    current_length = 0

    for word in words:
        extra_space = 1 if current_words else 0
        projected_length = current_length + extra_space + len(word)

        if current_words and projected_length > chunk_size:
            chunks.append(" ".join(current_words))
            current_words = [word]
            current_length = len(word)
        else:
            current_words.append(word)
            current_length = projected_length

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks
