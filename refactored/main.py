import argparse
import logging
import sys

from config import setup_logging, get_api_key
from document_processor import load_documents, process_documents, rerank_results
from vector_store import create_vector_store, get_relevant_documents
from chat_manager import get_chat_response, format_sources, get_mock_response
from utils import print_welcome_message, is_exit_command, format_response

# Toggle between mock and real responses
USE_MOCK = True

def test_embeddings(vector_store):
    """Test embedding quality with specific queries"""
    test_queries = [
        "What is FastAPI?",
        "Show me the main application setup",
        "How are routes defined?",
        "Error handling in FastAPI"
    ]

    logging.info("\n=== Testing Embedding Quality ===")
    for query in test_queries:
        docs = vector_store.similarity_search(query, k=3)
        logging.info(f"\nQuery: {query}")
        for i, doc in enumerate(docs, 1):
            logging.info(f"Match {i}: {doc.metadata['source']}")
            logging.info(f"Preview: {doc.page_content[:100]}...")

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description='Document QA System')
    parser.add_argument('--docs_dir', type=str, required=True, help='Directory containing documents')
    args = parser.parse_args()


    get_api_key()

    documents = load_documents(args.docs_dir)
    logging.info("=== Document Analysis ===")
    logging.info(f"Python files: {sum(1 for doc in documents if doc.metadata['source'].endswith('.py'))}")
    logging.info(f"Other files: {sum(1 for doc in documents if not doc.metadata['source'].endswith('.py'))}")

    chunks = process_documents(documents)
    logging.info("\n=== Chunk Analysis ===")
    for i, chunk in enumerate(chunks[:5]):
        logging.info(f"\nChunk {i+1}:")
        logging.info(f"File: {chunk.metadata['source']}")
        logging.info(f"Size: {len(chunk.page_content)} characters")
        logging.info(f"Preview:\n{chunk.page_content[:200]}...")

    logging.info("\nAnalysis complete - exiting")
    logging.info("\n=== Creating FAISS Index ===")
    vector_store = create_vector_store(chunks, args.docs_dir)
    logging.info("FAISS index creation complete")
    test_embeddings(vector_store)

if __name__ == "__main__":
    main()
