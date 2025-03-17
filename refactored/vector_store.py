import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
import logging
import time
from langchain_huggingface import HuggingFaceEmbeddings
import json

MODELS_CACHE_DIR = "models_cache"

FAISS_INDEX_PATH = "faiss_hf_index"
FAISS_TIMESTAMP_FILE = "faiss_timestamp.txt"
CHUNK_STATS_PATH = "faiss_hf_index/chunk_stats.json"
CODE_INDEX_PATH = "faiss_hf_index/code"
DOCS_INDEX_PATH = "faiss_hf_index/docs"


def save_chunk_stats(chunks):
    # Filter for unique, non-test files
    unique_files = set()
    selected_chunks = []

    for chunk in chunks:
        file_path = chunk.metadata['source']
        if (not 'test' in file_path.lower() and
            file_path not in unique_files and
            file_path.endswith('.py')):
            unique_files.add(file_path)
            selected_chunks.append(chunk)
            if len(selected_chunks) >= 5:
                break

    stats = {
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
            for chunk in selected_chunks
        ]
    }

    with open(CHUNK_STATS_PATH, 'w') as f:
        json.dump(stats, f)



def load_chunk_stats():
    """Load chunk statistics from JSON file"""
    with open(CHUNK_STATS_PATH, 'r') as f:
        return json.load(f)


def is_index_fresh(source_dir):
    """Check if FAISS index is up to date with source files"""
    if not os.path.exists(FAISS_TIMESTAMP_FILE):
        return False

    with open(FAISS_TIMESTAMP_FILE, 'r') as f:
        index_time = float(f.read().strip())

    # Get latest modification time of source files
    latest_mod = max(os.path.getmtime(os.path.join(root, file))
                    for root, _, files in os.walk(source_dir)
                    for file in files if file.endswith('.py'))

    return index_time > latest_mod

def save_index_timestamp():
    """Save current timestamp when index is created"""
    with open(FAISS_TIMESTAMP_FILE, 'w') as f:
        f.write(str(time.time()))

def get_mixed_documents(vector_store, query, k=6):
    """Retrieve a balanced mix of Python and Markdown files"""
    # Get a larger initial set to ensure we have both types
    initial_docs = vector_store.similarity_search(query, k=k*3)

    logging.info(f"Initial retrieval: {len(initial_docs)} documents")
    for doc in initial_docs:
        logging.info(f"Retrieved: {doc.metadata['source']}")

    # Separate by file type
    py_docs = [doc for doc in initial_docs if doc.metadata['source'].endswith('.py')]
    md_docs = [doc for doc in initial_docs if doc.metadata['source'].endswith('.md')]

    logging.info(f"Found {len(py_docs)} Python files and {len(md_docs)} Markdown files")

    # Take top k/2 from each type
    half_k = k // 2
    mixed_docs = py_docs[:half_k] + md_docs[:half_k]

    return mixed_docs


def get_relevant_documents(vector_store, query, k=4):
    """Retrieve only Python source files"""
    logging.info(f"Initial search for query: '{query}' with k={k}")

    # Get a larger initial set
    docs = vector_store.similarity_search(query, k=k*5)
    logging.info(f"Retrieved {len(docs)} initial documents")

    # Log initial results
    for idx, doc in enumerate(docs):
        logging.info(f"Initial result {idx+1}: {doc.metadata['source']} | Score: {doc.metadata.get('score', 'N/A')}")

    # Only keep Python files
    python_docs = [doc for doc in docs if doc.metadata['source'].endswith('.py')]
    logging.info(f"Filtered to {len(python_docs)} Python files")

    # If we don't have enough Python files, get more documents
    while len(python_docs) < k and k*5 < 100:
        k *= 2
        logging.info(f"Not enough Python files, expanding search to k={k*5}")
        docs = vector_store.similarity_search(query, k=k*5)
        python_docs = [doc for doc in docs if doc.metadata['source'].endswith('.py')]
        logging.info(f"Found {len(python_docs)} Python files after expansion")

    return python_docs[:k]
def test_embeddings_quality(vector_store):
    """Demonstrate embedding quality with targeted examples"""
    test_cases = [
        ("Show me FastAPI dependency injection", "Looking for core DI implementation"),
        ("How to handle authentication", "Seeking security related code"),
        ("FastAPI request lifecycle", "Finding middleware and core routing"),
        ("Database connection setup", "Database integration code")
    ]

    for query, purpose in test_cases:
        logging.info(f"\nTest Case: {purpose}")
        docs = vector_store.similarity_search(query, k=2)
        for doc in docs:
            logging.info(f"Found: {doc.metadata['source']}")
            logging.info(f"Relevance: {'High' if doc.metadata['source'].endswith('.py') else 'Medium'}")


def create_vector_store(chunks, source_dir):
    """Initialize embeddings with local caching"""
    start_time = time.time()
    logging.info("Starting embeddings initialization...")

    os.makedirs(MODELS_CACHE_DIR, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=MODELS_CACHE_DIR
    )

    if os.path.exists(FAISS_INDEX_PATH):
        logging.info(f"Found existing FAISS index at {FAISS_INDEX_PATH}")
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True  # Added this flag
        )
        logging.info(f"Loaded existing index in {time.time() - start_time:.2f} seconds")
    else:
        logging.info("Creating new FAISS index")
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)
        save_chunk_stats(chunks)
        logging.info(f"Created and saved new index in {time.time() - start_time:.2f} seconds")

    return vector_store






