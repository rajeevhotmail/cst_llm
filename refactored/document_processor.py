import os
from langchain_community.document_loaders import DirectoryLoader

import cohere
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_documents(documents):
    """Split documents using Python-aware chunking"""
    # Filter out empty or minimal content
    valid_docs = [doc for doc in documents if len(doc.page_content.strip()) > 50]

    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language="python",
        chunk_size=1500,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(valid_docs)
    logging.info(f"Split documents into {len(chunks)} Python-aware chunks")
    return chunks




def load_documents(directory_path):
    """Load documents in priority order"""
    loaders = [
        DirectoryLoader(directory_path, glob="**/*.py"),      # Python files first
        DirectoryLoader(directory_path, glob="**/README.*"),  # READMEs next
        DirectoryLoader(directory_path, glob="**/*.md"),      # Other docs
        DirectoryLoader(directory_path, glob="**/*.rst")      # ReStructuredText
    ]

    all_docs = []
    for loader in loaders:
        all_docs.extend(loader.load())

    logging.info(f"Loaded {len(all_docs)} documents by type")
    return all_docs



def rerank_results(retrieved_docs, query):
    """Return top 3 most relevant documents"""
    return retrieved_docs[:3]
    """Rerank retrieved documents using Cohere's reranking model."""
    co = cohere.Client(os.getenv("COHERE_API_KEY"))
    docs_text = [doc.page_content for doc in retrieved_docs]

    results = co.rerank(
        query=query,
        documents=docs_text,
        top_n=3,
        model='rerank-english-v2.0'
    )

    reranked_docs = []
    for result in results:
        index = result.index  # Direct access to index
        reranked_docs.append(retrieved_docs[index])

    return reranked_docs
