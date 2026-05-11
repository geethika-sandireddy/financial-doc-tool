import os
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from anomaly import detect_anomalies, extract_transactions
from embeddings import get_embedding, search_chunks
from pdf_processor import extract_text_from_pdf

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

chunks_store = []
embeddings_store = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global chunks_store, embeddings_store
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    chunks_store = extract_text_from_pdf(filepath)
    embeddings_store = [get_embedding(chunk['content']) for chunk in chunks_store]
    return jsonify({'message': f'Processed {len(chunks_store)} chunks from {filename}'})

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json(silent=True) or {}
    query = data.get('query', '')

    if not query:
        return jsonify({'error': 'No query provided'}), 400
    if not chunks_store:
        return jsonify({'error': 'No document uploaded yet'}), 400

    results = search_chunks(query, chunks_store, embeddings_store)
    return jsonify({'results': results})

@app.route('/anomalies', methods=['GET'])
def anomalies():
    if not chunks_store:
        return jsonify({'error': 'No document uploaded yet'}), 400
    transactions = extract_transactions(chunks_store)
    normal, flagged = detect_anomalies(transactions)
    return jsonify({
        'total_transactions': len(transactions),
        'normal': len(normal),
        'anomalies': flagged
    })

if __name__ == '__main__':
    app.run(debug=True)
