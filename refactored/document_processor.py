import os
import re
from langchain_community.document_loaders import DirectoryLoader
from langchain.schema import Document
import cohere
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter


def deduplicate_results(docs):
    """Ensure unique documents in results based on file path"""
    seen_paths = set()
    unique_docs = []

    for doc in docs:
        file_path = doc.metadata['source']
        if file_path not in seen_paths:
            seen_paths.add(file_path)
            unique_docs.append(doc)
            logging.info(f"✅ Keeping unique file: {file_path}")
        else:
            logging.info(f"🔄 Skipping duplicate: {file_path}")

    return unique_docs


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

def calculate_keyword_match(query: str, content: str) -> float:
    """Calculate exact keyword match score between query and content"""
    query_terms = query.lower().split()
    content_lower = content.lower()

    # Count matches
    matches = sum(1 for term in query_terms if term in content_lower)

    # Calculate score based on percentage of matching terms
    score = matches / len(query_terms) if query_terms else 0.0

    logging.info(f"Keyword match score: {score:.2f} ({matches} matches)")
    return score

def score_code_quality(doc: Document) -> float:
    """Score the code quality based on documentation and structure"""
    content = doc.page_content
    file_path = doc.metadata['source']

    # Base score
    quality_score = 0.5

    # Check for docstrings and comments
    if '"""' in content or "'''" in content:
        quality_score += 0.2
        logging.info(f"Docstring bonus: {file_path}")

    # Check for well-structured code indicators
    if 'def ' in content or 'class ' in content:
        quality_score += 0.2
        logging.info(f"Structure bonus: {file_path}")

    # Check for descriptive naming
    if re.search(r'[a-z]+_[a-z]+', content):
        quality_score += 0.1
        logging.info(f"Naming bonus: {file_path}")

    return min(1.0, quality_score)



def calculate_file_importance(file_path: str) -> float:
    """Calculate importance score of a file based on its path and role"""
    importance_score = 0.5  # Base score

    # Core functionality files get higher scores
    if '/core/' in file_path or 'main.py' in file_path:
        importance_score += 0.3
        logging.info(f"Core file bonus: {file_path}")

    # Files in root directory are often more important
    if file_path.count('/') <= 2:
        importance_score += 0.2
        logging.info(f"Root proximity bonus: {file_path}")

    # Key functionality indicators in path
    key_terms = ['api', 'model', 'service', 'controller']
    if any(term in file_path.lower() for term in key_terms):
        importance_score += 0.2
        logging.info(f"Key functionality bonus: {file_path}")

    # Normalize score to 0-1 range
    return min(1.0, importance_score)


def rerank_results(retrieved_docs, query):
    """Custom reranking of retrieved documents using multiple scoring factors"""
    logging.info(f"Starting custom reranking for query: '{query}'")

    # First deduplicate the input documents
    unique_docs = deduplicate_results(retrieved_docs)
    logging.info(f"Deduplication: {len(retrieved_docs)} -> {len(unique_docs)} documents")

    # Log initial documents
    for idx, doc in enumerate(unique_docs):
        logging.info(f"Pre-rerank {idx+1}: {doc.metadata['source']}")

    scored_docs = []
    for doc in unique_docs:
        # Calculate multiple scoring factors
        exact_match_score = calculate_keyword_match(query, doc.page_content)
        code_quality_score = score_code_quality(doc)
        file_importance_score = calculate_file_importance(doc.metadata['source'])

        final_score = (
            0.4 * exact_match_score +
            0.3 * code_quality_score +
            0.3 * file_importance_score
        )

        scored_docs.append((doc, final_score))
        logging.info(f"Scored {doc.metadata['source']}: match={exact_match_score:.2f}, quality={code_quality_score:.2f}, importance={file_importance_score:.2f}, final={final_score:.2f}")

    # Sort by final score
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # Get top 3 results
    reranked = [doc for doc, score in scored_docs[:3]]

    # Log final ranking
    logging.info("Final ranked results:")
    for idx, doc in enumerate(reranked):
        logging.info(f"Final rank {idx+1}: {doc.metadata['source']}")

    return reranked