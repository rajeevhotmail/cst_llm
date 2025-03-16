from flask import Flask, render_template, request, jsonify
from vector_store import create_vector_store
from document_processor import load_documents, process_documents
import logging
import time
from config import setup_logging
import os

app = Flask(__name__)
setup_logging()  # This will create the timestamped log file
start_total = time.time()



vector_store = None

def initialize_vector_store():
    global vector_store
    if vector_store is None:
        start_total = time.time()
        vector_store = create_vector_store(None, "source_code/fastapi")
        logging.info(f"Total initialization time: {time.time() - start_total:.2f} seconds")
    return vector_store

# Initialize once at startup
vector_store = initialize_vector_store()


if os.path.exists("faiss_hf_index"):
    t0 = time.time()
    vector_store = create_vector_store(None, "source_code/fastapi")
    logging.info(f"Loaded existing vector store in: {time.time() - t0:.2f} seconds")
else:
    t1 = time.time()
    docs = load_documents("source_code/fastapi")
    logging.info(f"Document loading took: {time.time() - t1:.2f} seconds")

    t2 = time.time()
    chunks = process_documents(docs)
    logging.info(f"Document processing took: {time.time() - t2:.2f} seconds")

    t3 = time.time()
    vector_store = create_vector_store(chunks, "source_code/fastapi")
    logging.info(f"Vector store creation took: {time.time() - t3:.2f} seconds")

logging.info(f"Total initialization time: {time.time() - start_total:.2f} seconds")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.json['query']
    results = vector_store.similarity_search(query, k=3)

    formatted_results = [{
        'source': doc.metadata['source'],
        'content': doc.page_content[:200],
        'relevance': 'High' if doc.metadata['source'].endswith('.py') else 'Medium'
    } for doc in results]

    return jsonify(formatted_results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)