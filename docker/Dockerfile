FROM python:3.12-slim

WORKDIR /app

# System deps for faiss-cpu / scikit-learn wheels build cleanly on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY templates ./templates

RUN pip install --no-cache-dir .

RUN mkdir -p uploads

# Default port for local runs / platforms (e.g. HF Spaces) that expect a
# fixed port. Render and similar platforms inject their own PORT at runtime,
# which the shell-form CMD below picks up automatically.
ENV PORT=7860
EXPOSE 7860

CMD gunicorn "financial_doc_tool.api.app:create_app()" --bind 0.0.0.0:${PORT} --workers 2 --timeout 120
