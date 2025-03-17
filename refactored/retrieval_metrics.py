from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from collections import deque
from typing import List, Dict

# Global storage for metrics
recent_search_metrics = deque(maxlen=5)  # Stores last 5 searches
search_times = deque(maxlen=100)  # Stores search response times
file_match_counts = {}  # Tracks which files match queries most often

def calculate_avg_response_time() -> float:
    return sum(search_times) / len(search_times) if search_times else 0.0

def get_top_matching_files() -> List[Dict]:
    return sorted(
        [{"file": k, "matches": v} for k, v in file_match_counts.items()],
        key=lambda x: x["matches"],
        reverse=True
    )[:5]

def compute_similarity_score(query, content):
    """Compute semantic similarity between query and content"""
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    query_embedding = model.encode([query])
    content_embedding = model.encode([content])

    similarity = cosine_similarity(query_embedding, content_embedding)[0][0]
    return float(similarity)

def calculate_relevance_metrics(query, results):
    """Calculate detailed relevance metrics for search results"""
    metrics = {
        "query": query,
        "total_results": len(results),
        "relevance_scores": [
            {
                "file": doc.metadata['source'],
                "score": compute_similarity_score(query, doc.page_content),
                "match_type": "Code" if doc.metadata['source'].endswith('.py') else "Documentation"
            }
            for doc in results
        ],
        "avg_score": 0.0  # We'll calculate this
    }

    if metrics["relevance_scores"]:
        metrics["avg_score"] = sum(r["score"] for r in metrics["relevance_scores"]) / len(metrics["relevance_scores"])

    return metrics
