from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
import logging


def create_vector_store(chunks):
    """Initialize FAISS vector store with document chunks."""
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    logging.info("Vector store created successfully")
    return vector_store

def get_relevant_documents(vector_store, query, k=4):
    """Retrieve relevant documents from vector store based on query."""
    docs = vector_store.similarity_search(query, k=k)
    logging.info(f"Retrieved {len(docs)} relevant documents for query")
    return docs
