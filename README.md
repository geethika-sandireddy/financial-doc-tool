# Financial Document Tool

Small Flask app for uploading a financial PDF, searching it with Gemini embeddings, and flagging unusual transaction amounts.

## What it does

- Extracts text from uploaded PDFs
- Breaks the document into chunks for semantic search
- Uses Gemini embeddings to find the most relevant passages for a query
- Runs a simple anomaly check on detected transaction amounts

## Setup

1. Create and activate a virtual environment if you want one.
2. Install the dependencies:

```bash
pip install flask google-generativeai numpy pandas pypdf python-dotenv scikit-learn
```

3. Create a local `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

4. Start the app:

```bash
python app.py
```

5. Open [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Notes

- `.env` is ignored by Git and should stay local.
- Uploaded files are stored in the `uploads/` folder.
