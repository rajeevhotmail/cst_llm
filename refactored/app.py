from flask import Flask, render_template, request, jsonify
from vector_store import create_vector_store, get_mixed_documents
from vector_store import load_chunk_stats
from document_processor import load_documents, process_documents, rerank_results
import logging
import time
from config import setup_logging
import os
from retrieval_metrics import (
    calculate_relevance_metrics,
    recent_search_metrics,
    search_times,
    file_match_counts,
    calculate_avg_response_time,
    get_top_matching_files
)

app = Flask(__name__)
setup_logging()  # This will create the timestamped log file
start_total = time.time()

# Load documents and create chunks first
chunks = None
vector_store = None

if os.path.exists("faiss_hf_index"):
    vector_store = create_vector_store(None, "source_code/fastapi")
    chunks_info = load_chunk_stats()
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

@app.route('/chunks-analysis')
def show_chunking_stats():
    if os.path.exists("faiss_hf_index"):
        # Use cached stats
        chunks_info = load_chunk_stats()
    else:
        # Generate stats from chunks
        chunks_info = {
            "total_chunks": len(chunks),
            "file_types": {
                "python": sum(1 for c in chunks if c.metadata['source'].endswith('.py')),
                "docs": sum(1 for c in chunks if c.metadata['source'].endswith('.md'))
            },
            "sample_chunks": [
                {
                    "source": chunk.metadata['source'],
                    "size": len(chunk.page_content),
                    "type": "Python" if chunk.metadata['source'].endswith('.py') else "Documentation"
                }
                for chunk in chunks[:5]
            ]
        }
    return jsonify(chunks_info)
@app.route('/search', methods=['POST'])
def search():
    query = request.json['query']
    start_time = time.time()

    logging.info(f"🔍 Processing search query: {query}")

    # Use our new mixed document retrieval
    initial_results = get_mixed_documents(vector_store, query, k=6)
    logging.info(f"📑 Retrieved {len(initial_results)} mixed results")

    # Apply reranking
    reranked_results = rerank_results(initial_results, query)
    logging.info(f"🏆 Reranked to {len(reranked_results)} final results")

    search_time = time.time() - start_time
    logging.info(f"⏱️ Total search time: {search_time:.3f} seconds")

    # Calculate and store metrics
    search_metrics = calculate_relevance_metrics(query, reranked_results)
    recent_search_metrics.append(search_metrics)
    search_times.append(search_time)

    logging.info(f"📊 Metrics collected: {search_metrics}")

    # Update file match counts
    for doc in reranked_results:
        file_match_counts[doc.metadata['source']] = file_match_counts.get(doc.metadata['source'], 0) + 1

    formatted_results = [{
        'source': doc.metadata['source'],
        'content': doc.page_content[:200],
        'relevance': 'High' if doc.metadata['source'].endswith('.py') else 'Medium',
        'similarity_score': search_metrics['relevance_scores'][idx]['score']
    } for idx, doc in enumerate(reranked_results)]

    return jsonify(formatted_results)

@app.route('/retrieval-analysis')
def show_retrieval_stats():
    metrics_info = {
        "recent_searches": list(recent_search_metrics)[-5:],  # Convert deque to list for slicing
        "avg_response_time": calculate_avg_response_time(),
        "top_matching_files": get_top_matching_files()
    }
    return jsonify(metrics_info)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)