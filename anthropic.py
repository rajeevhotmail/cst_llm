from flask import Flask, request, jsonify, render_template
import os
import faiss
import pickle
import torch
import tree_sitter
from tree_sitter import Language, Parser
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic
from dotenv import load_dotenv

# Load the .env file
load_dotenv("anthropic_key.env")
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Configuration
PYTHON_FILES_DIR = "./repo"  # Directory containing Python files
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FAISS_CST_INDEX_FILE = "faiss_cst.idx"
FAISS_TEXT_INDEX_FILE = "faiss_text.idx"
CODE_SNIPPETS_FILE = "code_snippets.pkl"
README_FILE = "README.md"

# Load Sentence Transformer model
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Load Tree-sitter parser
PYTHON_LANGUAGE = Language("build/my-languages.so", "python")
parser = Parser()
parser.set_language(PYTHON_LANGUAGE)

# FAISS Setup
dimension = 384
faiss_cst_index = faiss.IndexFlatL2(dimension)
faiss_text_index = faiss.IndexFlatL2(dimension)

# Store code snippets
code_snippets = []
readme_text = ""

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
if os.path.exists(FAISS_CST_INDEX_FILE) and os.path.exists(FAISS_TEXT_INDEX_FILE) and os.path.exists(CODE_SNIPPETS_FILE):
    print("✅ Loading precomputed FAISS index and code snippets...")
    faiss_cst_index = faiss.read_index(FAISS_CST_INDEX_FILE)
    faiss_text_index = faiss.read_index(FAISS_TEXT_INDEX_FILE)
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
                    cst_embedding = embedding_model.encode(cst_text, convert_to_numpy=True)
                    text_embedding = embedding_model.encode(source_code, convert_to_numpy=True)
                    faiss_cst_index.add(cst_embedding.reshape(1, -1))
                    faiss_text_index.add(text_embedding.reshape(1, -1))
                    code_snippets.append((filepath, source_code))

    # Save FAISS Index and Snippets
    faiss.write_index(faiss_cst_index, FAISS_CST_INDEX_FILE)
    faiss.write_index(faiss_text_index, FAISS_TEXT_INDEX_FILE)
    with open(CODE_SNIPPETS_FILE, "wb") as f:
        pickle.dump(code_snippets, f)

# Load README if available
if os.path.exists(README_FILE):
    with open(README_FILE, "r", encoding="utf-8") as f:
        readme_text = f.read()
        print("✅ README.md loaded for project summary!")

print(f"Total Python files processed: {len(code_snippets)}")
print(f"FAISS CST Index Size: {faiss_cst_index.ntotal}")
print(f"FAISS Text Index Size: {faiss_text_index.ntotal}")

def retrieve_project_summary():
    """Returns a structured project-level summary using README, docstrings, or file content."""
    if readme_text:
        print("📖 Returning README-based project summary.")
        return readme_text[:1000] + "..."

    summary_lines = []
    for filepath, source_code in code_snippets[:5]:
        filename = os.path.basename(filepath)
        lines = source_code.split("\n")

        # Try to find module docstring
        docstring = ""
        for line in lines:
            line = line.strip()
            if line.startswith('"""') or line.startswith("'''"):
                docstring = line.strip('"').strip("'").strip()
                break

        # If no docstring, look for key imports and class/function definitions
        if not docstring:
            key_lines = []
            for line in lines[:10]:  # Look at first 10 lines
                if line.strip().startswith(('import ', 'from ', 'class ', 'def ')):
                    key_lines.append(line.strip())
            if key_lines:
                docstring = "Contains: " + "; ".join(key_lines[:3])  # Show first 3 key elements

        summary_lines.append(f"📂 **{filename}**\n   {docstring if docstring else 'Python module file'}")

    if summary_lines:
        print("📜 Returning file-based summary.")
        return "\n\n".join(summary_lines)

    return "No project summary available."

def retrieve_code_snippets(query):
    """Retrieves the most relevant code snippet based on query."""
    if len(code_snippets) == 0:
        return None, "No code snippets available."

    query_embedding = embedding_model.encode(query, convert_to_numpy=True)
    distances_cst, indices_cst = faiss_cst_index.search(query_embedding.reshape(1, -1), 3)
    distances_text, indices_text = faiss_text_index.search(query_embedding.reshape(1, -1), 3)

    print(f"🔍 Query: {query}")
    print(f"📏 CST Distances: {distances_cst}")
    print(f"📏 Text Distances: {distances_text}")

    THRESHOLD = 1.7
    filtered_indices_cst = [i for i, d in zip(indices_cst[0], distances_cst[0]) if d < THRESHOLD]
    filtered_indices_text = [i for i, d in zip(indices_text[0], distances_text[0]) if d < THRESHOLD]

    if filtered_indices_cst:
        print("✅ Using CST-based retrieval")
        return code_snippets[filtered_indices_cst[0]]
    elif filtered_indices_text:
        print("⚠ CST retrieval failed, falling back to raw text")
        return code_snippets[filtered_indices_text[0]]
    else:
        print("🌍 No relevant local match, querying internet...")
        return None, "Querying external sources."

def query_claude(question):
    """Queries Claude API with retrieved code snippet or project details."""
    result = retrieve_code_snippets(question)
    if result is None or result[0] is None:
        return "No relevant code snippet found."

    filename, snippet = result
    extracted_cst = extract_cst_features(snippet)

    prompt = f"""
    The following is a Python code snippet from `{filename}`:
    ```python
    {snippet}
    ```

    The extracted syntax structure of the code (CST) is:
    ```text
    {extracted_cst}
    ```

    Please analyze and explain this code, considering both its structure and functionality.
    Question: {question}
    """

    try:
        response = anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return response.content[0].text
    except Exception as e:
        print(f"Error querying Claude: {str(e)}")
        return f"Error: {str(e)}"

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def handle_query():
    data = request.json
    question = data.get("question", "").lower()

    if not question:
        return jsonify({"error": "No question provided"}), 400

    project_keywords = ["project overview", "what is the project about", "workflow"]
    if any(kw in question for kw in project_keywords):
        print("📖 Project-level or workflow question detected.")
        answer = retrieve_project_summary()
    else:
        print("🔍 Code-related query detected, retrieving from FAISS.")
        answer = query_claude(question)

    return jsonify({"question": question, "answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)