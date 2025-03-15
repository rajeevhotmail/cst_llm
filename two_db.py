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
    """Returns a structured project-level summary using README, docstrings, or CST."""

    # ✅ 1. Use README if available
    if readme_text:
        print("📖 Returning README-based project summary.")
        return readme_text[:1000] + "..."  # Limit to first 1000 chars

    # ✅ 2. Extract high-level workflow from Python docstrings
    workflow_summary = []
    for filepath, source_code in code_snippets[:5]:  # Limit to first 5 files
        lines = source_code.split("\n")
        docstrings = [line.strip() for line in lines if line.strip().startswith('"""')]

        if docstrings:
            filename = os.path.basename(filepath)
            workflow_summary.append(f"📂 **{filename}** → {docstrings[0]}")  # Use first docstring

    if workflow_summary:
        print("📜 Returning workflow summary from docstrings.")
        return "\n".join(workflow_summary)

    # ✅ 3. Extract workflow structure from CST if no docstrings found
    if code_snippets:
        print("📄 No README/docstrings found, summarizing CST.")
        summary_lines = []
        for snippet in code_snippets[:5]:  # Limit to 5 files
            filename = os.path.basename(snippet[0])
            cst_features = extract_cst_features(snippet[1]).split()[:10]  # Limit CST tokens
            summary_lines.append(f"📂 **{filename}** → CST Elements: {' '.join(cst_features)}")

        return "\n".join(summary_lines)

    return "No project summary available."


def retrieve_code_snippets(query):
    """Retrieves the most relevant code snippet based on query, prioritizing CST but falling back to raw text."""
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

def query_chatgpt(question):
    """Queries OpenAI GPT-4o with retrieved code snippet or project details."""

    # ✅ Check if the query is asking for the workflow
    if "workflow" in question.lower():
        print("📌 Detected workflow-related query.")
        return retrieve_project_summary()

    # ✅ Retrieve relevant code snippet
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

    Explain the code using both its raw structure and functional purpose.
    Question: {question}
    """

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are an AI that explains Python code."},
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
    question = data.get("question", "").lower()

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # ✅ Detect project-level or workflow-related questions
    project_keywords = ["project overview", "what is the project about", "workflow"]
    if any(kw in question for kw in project_keywords):
        print("📖 Project-level or workflow question detected.")
        answer = retrieve_project_summary()
    else:
        print("🔍 Code-related query detected, retrieving from FAISS.")
        answer = query_chatgpt(question)

    return jsonify({"question": question, "answer": answer})




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
