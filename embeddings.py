import os
import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_embedding(text):
    result = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return np.array(result['embedding'])

def get_query_embedding(text):
    result = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_query"
    )
    return np.array(result['embedding'])

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    magnitude = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if magnitude == 0:
        return 0
    return dot_product / magnitude

def search_chunks(query, chunks, embeddings, top_k=5):
    query_embedding = get_query_embedding(query)
    scores = []

    for index, embedding in enumerate(embeddings):
        score = cosine_similarity(query_embedding, embedding)
        scores.append((index, score))

    scores.sort(key=lambda x: x[1], reverse=True)

    results = []

    for index, score in scores[:top_k]:
        results.append({
            'content': chunks[index]['content'],
            'page': chunks[index]['page'],
            'source': chunks[index]['source'],
            'score': round(float(score), 4)
        })

    return results
