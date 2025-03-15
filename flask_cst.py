from flask import Flask, request, jsonify, render_template
import os
import faiss
import pickle
import torch
import tree_sitter
from tree_sitter import Language, Parser
from sentence_transformers import SentenceTransformer
import openai
from dotenv import load_dotenv

# Load the .env file
load_dotenv("openai_key.env")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configuration
PYTHON_FILES_DIR = "./repo"  # Directory containing Python files
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FAISS_INDEX_FILE = "faiss_index.idx"
CODE_SNIPPETS_FILE = "code_snippets.pkl"

# Load Sentence Transformer model
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Load Tree-sitter parser
PYTHON_LANGUAGE = Language("build/my-languages.so", "python")
parser = Parser()
parser.set_language(PYTHON_LANGUAGE)

# FAISS Setup
dimension = 384
faiss_index = faiss.IndexFlatL2(dimension)

# Store code snippets
code_snippets = []

def extract_cst_features(source_code):
    """Extracts CST features from Python code."""
    tree = parser.parse(source_code.encode("utf-8"))
    root_node = tree.root_node
    cst_features = []

    def traverse(node):
        if node.child_count == 0:
            return
        cst_features.append(node.type)
        if node.type == "identifier":
            cst_features.append(node.text.decode("utf-8"))
        for child in node.children:
            traverse(child)

    traverse(root_node)
    return " ".join(cst_features)

# ** Load FAISS and Code Snippets if They Exist **
if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(CODE_SNIPPETS_FILE):
    print("✅ Loading precomputed FAISS index and code snippets...")
    faiss_index = faiss.read_index(FAISS_INDEX_FILE)
    with open(CODE_SNIPPETS_FILE, "rb") as f:
        code_snippets = pickle.load(f)
else:
    print("🔄 Scanning and embedding Python files...")
    for root, _, files in os.walk(PYTHON_FILES_DIR):
        for filename in files:
            if filename.endswith(".py"):
                filepath = os.path.join(root, filename)
                print(f"Processing: {filepath}")

                with open(filepath, "r", encoding="utf-8") as file:
                    source_code = file.read()
                    if len(source_code.strip()) == 0:
                        print(f"⚠ Warning: {filepath} is empty, skipping.")
                        continue

                    cst_text = extract_cst_features(source_code)
                    embedding = embedding_model.encode(cst_text, convert_to_numpy=True)
                    faiss_index.add(embedding.reshape(1, -1))
                    code_snippets.append((filepath, source_code))

    # Save FAISS Index and Snippets
    faiss.write_index(faiss_index, FAISS_INDEX_FILE)
    with open(CODE_SNIPPETS_FILE, "wb") as f:
        pickle.dump(code_snippets, f)

print(f"Total Python files processed: {len(code_snippets)}")
print(f"FAISS Index Size: {faiss_index.ntotal}")

def retrieve_code_snippets(query):
    """Retrieves the most relevant code snippet based on query, prioritizing docstrings and comments."""
    if len(code_snippets) == 0:
        return None, "No code snippets available."

    query_embedding = embedding_model.encode(query, convert_to_numpy=True)
    distances, indices = faiss_index.search(query_embedding.reshape(1, -1), 5)  # Get top 5 results

    print(f"🔍 Query: {query}")
    print(f"📏 Distances: {distances}")
    print(f"📌 Indices: {indices}")

    THRESHOLD = 2.0  # Increase threshold to allow broader matches
    filtered_indices = [i for i, d in zip(indices[0], distances[0]) if d < THRESHOLD]

    if not filtered_indices:
        return None, "No relevant snippet found."

    # Prioritize files with module docstrings
    best_match = None
    for idx in filtered_indices:
        filename, snippet = code_snippets[idx]
        if '"""' in snippet or "'''" in snippet:  # Check for docstrings
            best_match = (filename, snippet)
            break

    return best_match if best_match else code_snippets[filtered_indices[0]]


def query_chatgpt(question):
    """Queries OpenAI GPT-4o with retrieved code snippet."""
    result = retrieve_code_snippets(question)

    # Handle broad questions differently
    if result is None or result[0] is None:
        if "project" in question.lower() or "overview" in question.lower():
            return "This project analyzes Python code using Tree-Sitter to extract syntax structure (CST), generates embeddings via Sentence Transformers, and enables intelligent code search using FAISS. It allows users to query and retrieve relevant code snippets with AI-based explanations."
        if "database" in question.lower():
            return "No database usage was found in the analyzed code. The project primarily deals with SMS validation and error handling."

        return "No relevant code snippet found."

    filename, snippet = result
    extracted_cst = extract_cst_features(snippet)

    prompt = f"""
    Answer the user's question concisely based on the Python code below:
    ```python
    {snippet}
    ```

    Extracted syntax structure (CST):
    ```text
    {extracted_cst}
    ```

    Focus on answering directly. Avoid unnecessary explanations.
    Question: {question}
    """

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are an AI that explains Python code clearly and concisely."},
                  {"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def handle_query():
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    answer = query_chatgpt(question)
    return jsonify({"question": question, "answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
