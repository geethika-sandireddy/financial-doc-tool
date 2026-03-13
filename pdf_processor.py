import pypdf
import os

def extract_text_from_pdf(pdf_path):
    text_chunks = []
    with open(pdf_path, 'rb') as file:
        reader = pypdf.PdfReader(file)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():
                chunks = chunk_text(text, chunk_size=500)
                for chunk in chunks:
                    text_chunks.append({
                        'content': chunk,
                        'page': page_num + 1,
                        'source': os.path.basename(pdf_path)
                    })
    return text_chunks

def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    for word in words:
        current_chunk.append(word)
        current_size += len(word)
        if current_size >= chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks