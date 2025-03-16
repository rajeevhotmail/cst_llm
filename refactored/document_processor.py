from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import cohere
import logging

def load_documents(directory_path):
    """Load all text documents from the specified directory."""
    loader = DirectoryLoader(directory_path, glob="**/*.txt")
    documents = loader.load()
    logging.info(f"Loaded {len(documents)} documents from {directory_path}")
    return documents

def process_documents(documents):
    """Split documents into chunks and create embeddings."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Split documents into {len(chunks)} chunks")
    return chunks

def rerank_results(retrieved_docs, query):
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
        reranked_docs.append(retrieved_docs[result.index])

    return reranked_docs
