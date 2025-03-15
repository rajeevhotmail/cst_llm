import os
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import pickle
import logging
from tree_sitter import Language, Parser
import re

logging.basicConfig(level=logging.INFO)

# Configuration
BASE_DIR = os.getcwd()
PYTHON_FILES_DIR = os.path.join(BASE_DIR, "source_code")
FAISS_DB_DIR = os.path.join(BASE_DIR, "faiss_simplified_db")
FAISS_INDEX_FILE = os.path.join(FAISS_DB_DIR, "faiss_index.idx")
CODE_SNIPPETS_FILE = os.path.join(FAISS_DB_DIR, "code_snippets.pkl")

# Ensure FAISS directory exists
os.makedirs(FAISS_DB_DIR, exist_ok=True)

# Initialize Flask App
app = Flask(__name__)

# Load CodeBERT model for embeddings
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

# Load Tree-sitter parser
PYTHON_LANGUAGE = Language("build/my-languages.so", "python")
parser = Parser()
parser.set_language(PYTHON_LANGUAGE)

# FAISS Setup
dimension = 768  # CodeBERT dimension
faiss_index = faiss.IndexFlatL2(dimension)
code_snippets = []

# Generate embeddings using CodeBERT
def get_embedding(text):
    """Generate embeddings using CodeBERT model."""
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings.numpy()

# Extract function and class definitions from a Python file
def extract_functions_and_classes(filepath, source_code):
    """Extracts function and class definitions from Python code."""
    tree = parser.parse(source_code.encode("utf-8"))
    root_node = tree.root_node
    source_lines = source_code.splitlines()
    functions_and_classes = []

    def get_node_text(node):
        start_point, end_point = node.start_point, node.end_point
        if start_point[0] == end_point[0]:
            return source_lines[start_point[0]][start_point[1]:end_point[1]]
        else:
            result = [source_lines[start_point[0]][start_point[1]:]]
            for i in range(start_point[0] + 1, end_point[0]):
                result.append(source_lines[i])
            result.append(source_lines[end_point[0]][:end_point[1]])
            return '\n'.join(result)

    def get_definition_type(node):
        if node.type == "function_definition":
            for child in node.children:
                if child.type == "identifier":
                    return "function", child.text.decode("utf-8")
        elif node.type == "class_definition":
            for child in node.children:
                if child.type == "identifier":
                    return "class", child.text.decode("utf-8")
        return None, None

    def traverse_for_definitions(node):
        if node.type in ["function_definition", "class_definition"]:
            def_type, name = get_definition_type(node)
            if def_type and name:
                code_text = get_node_text(node)
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1

                functions_and_classes.append({
                    "type": def_type,
                    "name": name,
                    "code": code_text,
                    "start_line": start_line,
                    "end_line": end_line,
                    "filepath": filepath
                })
                logging.info(f"Extracted {def_type}: {name} from {filepath}")

        for child in node.children:
            traverse_for_definitions(child)

    traverse_for_definitions(root_node)
    return functions_and_classes

# Check and initialize FAISS index if missing
if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(CODE_SNIPPETS_FILE):
    print("✅ Loading precomputed FAISS index...")
    faiss_index = faiss.read_index(FAISS_INDEX_FILE)
    with open(CODE_SNIPPETS_FILE, "rb") as f:
        code_snippets = pickle.load(f)
    print(f"✅ Loaded {len(code_snippets)} code snippets")
else:
    print("🔄 No FAISS index found. Initializing from scratch...")
    for root, _, files in os.walk(PYTHON_FILES_DIR):
        for filename in files:
            if filename.endswith(".py"):
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as file:
                        source_code = file.read()
                        if not source_code.strip():
                            continue

                        # Extract functions and classes
                        extracted_items = extract_functions_and_classes(filepath, source_code)
                        print(f"📄 Extracted {len(extracted_items)} functions/classes from {filepath}")

                        for item in extracted_items:
                            code = item["code"]

                            # Create embedding
                            embedding = get_embedding(code)
                            faiss_index.add(embedding)

                            # Add to code snippets
                            code_snippets.append(item)
                except Exception as e:
                    print(f"❌ Error processing {filepath}: {str(e)}")

    print(f"✅ Indexed {len(code_snippets)} functions and classes")
    faiss.write_index(faiss_index, FAISS_INDEX_FILE)
    with open(CODE_SNIPPETS_FILE, "wb") as f:
        pickle.dump(code_snippets, f)
def rank_and_filter_snippets(retrieved_items, query, max_snippets=5):
    """
    Ranks and filters code snippets based on relevance criteria.

    Args:
        retrieved_items: List of code snippets from FAISS retrieval
        query: Original user query
        max_snippets: Maximum number of snippets to return

    Returns:
        List of filtered and ranked snippets
    """
    scored_items = []

    # Define important keywords for API-related queries
    api_keywords = ["api", "endpoint", "route", "request", "response", "handler", "function", "method"]

    # Check if query is API-related
    is_api_query = any(keyword in query.lower() for keyword in api_keywords)

    for item in retrieved_items:
        # Start with base score from embedding similarity (assumed to be pre-sorted)
        score = 100

        # Penalize test files
        if "/test" in item["filepath"] or "test_" in item["filepath"]:
            score -= 40

        # Penalize empty or very short functions
        if item["code"].count("\n") < 3:
            score -= 30

        # Reward files that match query keywords
        query_terms = query.lower().split()
        matching_terms = [term for term in query_terms if term in item["filepath"].lower() or term in item["code"].lower()]
        score += len(matching_terms) * 10

        # Prioritize functions/classes with docstrings
        if '"""' in item["code"] or "'''" in item["code"]:
            score += 25

        # For API queries, prioritize route handlers and API-related code
        if is_api_query:
            if "@app.route" in item["code"] or "@blueprint.route" in item["code"]:
                score += 50
            elif any(keyword in item["code"].lower() for keyword in api_keywords):
                score += 30

        # Check if function has parameters (sign of being an actual API)
        if item["type"] == "function" and "def " in item["code"]:
            params = item["code"].split("def ")[1].split("(")[1].split(")")[0]
            if params.strip() and params != "self":
                score += 15

        scored_items.append((score, item))

    # Sort by score (descending)
    scored_items.sort(reverse=True, key=lambda x: x[0])

    # Filter to top N items
    filtered_items = [item for _, item in scored_items[:max_snippets]]

    # Log the ranking process
    for score, item in scored_items[:max_snippets]:
        print(f"📊 Selected: {item['type']} {item['name']} (Score: {score})")

    if scored_items[max_snippets:]:
        print(f"❌ Filtered out {len(scored_items) - max_snippets} lower-ranked items")

    return filtered_items
def retrieve_code_snippets(query, top_k=5):  # Reduced from 20 to 5 for simplicity
    """Retrieves the most relevant code snippets based on query."""
    if len(code_snippets) == 0:
        return "No code snippets available."

    query_embedding = get_embedding(query)
    distances, indices = faiss_index.search(query_embedding, top_k)

    print(f"🔍 Query: {query}")
    print(f"📏 Distances: {distances}")

    retrieved_items = []
    for idx in indices[0]:
        if idx < len(code_snippets):
            retrieved_items.append(code_snippets[idx])

    if not retrieved_items:
        return "No relevant code found."

     # Apply ranking and filtering
    filtered_items = rank_and_filter_snippets(retrieved_items, query, max_snippets=5)

     # Format the retrieved snippets into a context string
    context_parts = []
    for snippet in filtered_items:
        snippet_type = snippet["type"]
        name = snippet["name"]
        code = snippet["code"]
        filepath = snippet["filepath"]

        header = f"{snippet_type.capitalize()}: {name} from {filepath} (lines {snippet['start_line']}-{snippet['end_line']})"
        context_parts.append(f"{header}\n```python\n{code}\n```\n")

    return "\n".join(context_parts)

# Simple mock response for testing (no LLM required)
def get_mock_response(question, context):
    return f"""
## Code Analysis Results

I analyzed the code related to your question: "{question}"

Here's what I found in the codebase:

{context}

This is a simplified response for testing purposes. In production, you would use an LLM to generate more detailed analysis.
"""

@app.route("/query", methods=["POST"])
def handle_query():
    try:
        data = request.json
        question = data.get("question", "")

        if not question:
            return jsonify({"error": "No question provided"}), 400
        print(f"📝 Received question: {question}")

        # Get relevant code snippets
        context = retrieve_code_snippets(question)
        print(f"📄 Retrieved relevant code snippets")

        # Get a mock response (replace with actual LLM in production)
        response = get_mock_response(question, context)

        return jsonify({"question": question, "answer": response})
    except Exception as e:
        import traceback
        print(f"❌ Server Error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)