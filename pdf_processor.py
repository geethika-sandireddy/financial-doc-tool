import os

import pypdf


def extract_text_from_pdf(pdf_path: str) -> list[dict[str, str | int]]:
    """Extract text from a PDF and return chunk metadata."""
    text_chunks: list[dict[str, str | int]] = []

    with open(pdf_path, "rb") as file:
        reader = pypdf.PdfReader(file)

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                for chunk in chunk_text(text, chunk_size=500):
                    text_chunks.append(
                        {
                            "content": chunk,
                            "page": page_num,
                            "source": os.path.basename(pdf_path),
                        }
                    )

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
