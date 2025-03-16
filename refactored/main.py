import argparse
import logging
from config import setup_logging, get_api_key
from document_processor import load_documents, process_documents, rerank_results
from vector_store import create_vector_store, get_relevant_documents
from chat_manager import get_chat_response, format_sources
from utils import print_welcome_message, is_exit_command, format_response

def main():
    parser = argparse.ArgumentParser(description='Document QA System')
    parser.add_argument('--docs_dir', type=str, required=True, help='Directory containing documents')
    args = parser.parse_args()

    logger = setup_logging()
    get_api_key()  # Verify API key exists

    # Load and process documents
    documents = load_documents(args.docs_dir)
    chunks = process_documents(documents)
    vector_store = create_vector_store(chunks)

    print_welcome_message()

    while True:
        query = input("\nQuestion: ").strip()

        if is_exit_command(query):
            print("Goodbye!")
            break

        # Get relevant documents and generate response
        relevant_docs = get_relevant_documents(vector_store, query)
        reranked_docs = rerank_results(relevant_docs, query)
        response = get_chat_response(query, reranked_docs)
        sources = format_sources(reranked_docs)

        print(format_response(response, sources))

if __name__ == "__main__":
    main()
