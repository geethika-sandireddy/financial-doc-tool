# Financial Document Intelligence

[![CI](https://github.com/geethika-sandireddy/financial-doc-tool/actions/workflows/ci.yml/badge.svg)](https://github.com/geethika-sandireddy/financial-doc-tool/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)

Detects anomalous transactions in financial PDFs by combining semantic
search (FAISS + Gemini embeddings) with unsupervised outlier detection
(Isolation Forest) over amounts extracted from unstructured document
text — not just a PDF chatbot.

**Measured, not claimed:** FAISS search is 27-30x faster than a brute-force
scan at 10K-50K chunks, and the anomaly detector's default threshold is
chosen from a precision/recall sweep against a labeled evaluation set, not
a guess. Full numbers in [`benchmarks/results.md`](benchmarks/results.md).

> No live demo is deployed yet — see [Deployment](#deployment) for how to run it with Docker.

## Architecture

```
                  ┌───────────────────────┐
  browser / curl ▶│  Flask (api/routes.py) │
                  │  session-scoped state  │
                  └───────────┬───────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
 core/pdf_processor.py  core/embeddings.py    core/anomaly.py
 parse + chunk PDF      batched Gemini calls   regex extraction +
 (page-count guard,     → google-genai SDK     IsolationForest,
 corrupted-file guard)                          tunable contamination
        │                     │
        └──────────┬──────────┘
                    ▼
          core/vector_store.py
          FAISS IndexFlatIP
          (exact cosine search,
           27-30x faster than the
           original Python loop
           at 10K+ chunks)
```

## What it does

- Uploads and parses financial PDFs (with page-count and corrupted-file guards)
- Chunks document text and embeds it in batches via the Gemini embeddings API
- Indexes chunks in FAISS for fast semantic search over natural-language queries
- Extracts currency-like values from the document text
- Flags unusual amounts with Isolation Forest, with a contamination default chosen from a measured precision/recall sweep, not a guess

## Tech stack

- Python 3.12, Flask, gunicorn
- Google Gemini embeddings (`google-genai` SDK)
- FAISS (`faiss-cpu`) for vector search
- scikit-learn (Isolation Forest), pandas, NumPy
- pypdf for PDF parsing
- pytest + pytest-cov + ruff, run in CI on every push

## Project structure

```
src/financial_doc_tool/
├── api/
│   ├── app.py          # Flask app factory
│   └── routes.py        # HTTP layer only — delegates to core/
├── core/
│   ├── pdf_processor.py # parsing + chunking
│   ├── embeddings.py    # batched Gemini embedding calls
│   ├── vector_store.py  # FAISS-backed search
│   └── anomaly.py       # transaction extraction + outlier detection
├── config.py             # centralized settings (env-driven)
└── exceptions.py

tests/
├── unit/         # core/ logic in isolation, no Flask, no network
├── integration/  # full upload → search → anomalies flow over real HTTP
└── performance/  # benchmarks + regression guards (search latency, anomaly precision/recall)

benchmarks/results.md  # measured numbers behind the claims above
docker/Dockerfile
```

## Setup

1. Install the package (editable, with dev dependencies):

```bash
pip install -e ".[dev]"
```

2. Copy the example environment file and add your Gemini API key:

```bash
cp .env.example .env
```

3. Start the dev server:

```bash
python run.py
```

4. Open [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Testing & CI

```bash
ruff check .
pytest --cov=financial_doc_tool --cov-report=term-missing
```

Every push runs lint and the full test suite (unit + integration +
performance regression guards) via GitHub Actions. Current coverage: 92%.

## Deployment

```bash
docker build -f docker/Dockerfile -t financial-doc-tool .
docker run -p 7860:7860 --env-file .env financial-doc-tool
```

The image runs on gunicorn (2 workers) rather than the Flask dev server.
Compatible with Hugging Face Spaces (Docker SDK) or any container host —
neither is currently deployed for this project.

## Design notes

- Gemini embeddings keep the search flow simple while giving strong semantic matching for natural-language queries; batching (see `core/embeddings.py`) turns N sequential API calls into `ceil(N / batch_size)`.
- FAISS's `IndexFlatIP` over L2-normalized vectors is mathematically identical to brute-force cosine similarity — no accuracy is traded for the speedup, only the Python-loop overhead is removed. See `benchmarks/results.md` for the measured numbers at different corpus sizes.
- Isolation Forest is a reasonable fit because anomaly detection here is unsupervised and the app doesn't assume labeled fraud data. Its `contamination` parameter is configurable via `.env` rather than hardcoded, and the default (0.1) is chosen from a precision/recall sweep in `benchmarks/results.md`, not an arbitrary guess.

## Limitations

- Session/document state is held in-process (per-session, in-memory); it does not survive a restart or scale across multiple app instances. A production deployment would move this to Redis or Postgres.
- The anomaly-detection evaluation uses synthetic data with one outlier "shape" (large-magnitude injected values) — see the caveat in `benchmarks/results.md`. It validates the detector's math and parameter choice; it is not a claim about accuracy on real, messy financial documents, where outliers can be subtler.
- Retrieval quality depends on PDF text-extraction quality; scanned/image-only PDFs with no text layer will not chunk usefully.
- Uploaded files are deleted from disk immediately after processing; only derived chunks/embeddings/vector index are kept in memory for the session.

## License

[MIT](LICENSE)
