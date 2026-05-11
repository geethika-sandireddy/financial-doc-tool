# Financial Document Tool

Financial Document Tool is a small Flask app for exploring financial PDFs with semantic search and lightweight anomaly detection. Upload a PDF, turn its pages into searchable chunks with Gemini embeddings, and flag unusual transaction amounts with Isolation Forest.

## What it does

- Uploads and reads financial PDF files
- Breaks document text into retrieval-friendly chunks
- Uses Gemini embeddings for semantic search
- Extracts currency-like values from the document
- Flags unusual amounts with unsupervised anomaly detection

## Tech stack

- Python
- Flask
- Google Gemini embeddings
- scikit-learn
- pandas
- NumPy
- pypdf

## Project structure

- [app.py](C:\Users\HP\OneDrive\Desktop\Git Projects\financial-doc-tool\app.py): Flask app and routes
- [pdf_processor.py](C:\Users\HP\OneDrive\Desktop\Git Projects\financial-doc-tool\pdf_processor.py): PDF parsing and chunking
- [embeddings.py](C:\Users\HP\OneDrive\Desktop\Git Projects\financial-doc-tool\embeddings.py): Gemini embedding helpers and similarity search
- [anomaly.py](C:\Users\HP\OneDrive\Desktop\Git Projects\financial-doc-tool\anomaly.py): transaction extraction and anomaly detection
- [templates/index.html](C:\Users\HP\OneDrive\Desktop\Git Projects\financial-doc-tool\templates\index.html): browser UI

## Setup

1. Create a virtual environment if you want one.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy the example environment file and add your Gemini API key:

```bash
copy .env.example .env
```

4. Set your local values in `.env`:

```env
GEMINI_API_KEY=your_api_key_here
FLASK_SECRET_KEY=replace_me_for_local_use
FLASK_DEBUG=true
```

5. Start the server:

```bash
python app.py
```

6. Open [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Notes

- The current app keeps document state in memory per browser session.
- Uploaded files are removed after processing.
- `.env` stays local and is ignored by Git.

## Limitations

- This is still a local single-process app, not a production deployment.
- Session data resets when the server restarts.
- Retrieval quality depends on PDF text extraction quality.

## Testing

Run the unit tests with:

```bash
pytest
```
