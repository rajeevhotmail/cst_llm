import os
import faiss
import pickle
import traceback
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from tree_sitter import Language, Parser
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Configuration
BASE_DIR = os.getcwd()
PYTHON_FILES_DIR = os.path.join(BASE_DIR, "source_code")
FAISS_DB_DIR = os.path.join(BASE_DIR, "faiss_db_startcoder")
FAISS_CST_INDEX_FILE = os.path.join(FAISS_DB_DIR, "faiss_cst.idx")
FAISS_TEXT_INDEX_FILE = os.path.join(FAISS_DB_DIR, "faiss_text.idx")
CODE_SNIPPETS_FILE = os.path.join(FAISS_DB_DIR, "code_snippets.pkl")

# Ensure FAISS directory exists
os.makedirs(FAISS_DB_DIR, exist_ok=True)

# Locate README dynamically
README_FILE = None
for root, dirs, files in os.walk(PYTHON_FILES_DIR):
    if "README.md" in files:
        README_FILE = os.path.join(root, "README.md")
        break

# Initialize Flask App
app = Flask(__name__)

# Load Sentence Transformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load Tree-sitter parser
PYTHON_LANGUAGE = Language("build/my-languages.so", "python")
parser = Parser()
parser.set_language(PYTHON_LANGUAGE)

# FAISS Setup
dimension = 384
faiss_cst_index = faiss.IndexFlatL2(dimension)
faiss_text_index = faiss.IndexFlatL2(dimension)
code_snippets = []
readme_text = ""

# Load StarCoder-Base-3B Model (lightweight and code-optimized)
model_name = "microsoft/phi-2"  # Use Phi-2 (lightweight, open-source)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    offload_folder="/mnt/data/huggingface_offload"  # Set offload location
)

pipeline_model = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Extract CST features
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

# Check and initialize FAISS indices if missing
if os.path.exists(FAISS_CST_INDEX_FILE) and os.path.exists(FAISS_TEXT_INDEX_FILE) and os.path.exists(CODE_SNIPPETS_FILE):
    print("✅ Loading precomputed FAISS CST and text indexes...")
    faiss_cst_index = faiss.read_index(FAISS_CST_INDEX_FILE)
    faiss_text_index = faiss.read_index(FAISS_TEXT_INDEX_FILE)
    with open(CODE_SNIPPETS_FILE, "rb") as f:
        code_snippets = pickle.load(f)
else:
    print("🔄 No FAISS indexes found. Initializing from scratch...")
    for root, _, files in os.walk(PYTHON_FILES_DIR):
        for filename in files:
            if filename.endswith(".py"):
                filepath = os.path.join(root, filename)
                with open(filepath, "r", encoding="utf-8") as file:
                    source_code = file.read()
                    if not source_code.strip():
                        continue
                    text_embedding = embedding_model.encode(source_code, convert_to_numpy=True)
                    faiss_text_index.add(text_embedding.reshape(1, -1))

                    cst_text = extract_cst_features(source_code)
                    cst_embedding = embedding_model.encode(cst_text, convert_to_numpy=True)
                    faiss_cst_index.add(cst_embedding.reshape(1, -1))

                    code_snippets.append((filepath, source_code))

    faiss.write_index(faiss_text_index, FAISS_TEXT_INDEX_FILE)
    faiss.write_index(faiss_cst_index, FAISS_CST_INDEX_FILE)
    with open(CODE_SNIPPETS_FILE, "wb") as f:
        pickle.dump(code_snippets, f)

# Load README if available
if README_FILE and os.path.exists(README_FILE):
    with open(README_FILE, "r", encoding="utf-8") as f:
        readme_text = f.read()
        print("✅ README.md loaded for project summary!")

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

    THRESHOLD = 2.0
    best_cst = distances_cst[0][0] if len(distances_cst[0]) > 0 else float('inf')
    best_text = distances_text[0][0] if len(distances_text[0]) > 0 else float('inf')

    if best_cst < THRESHOLD and best_text < THRESHOLD:
        if best_text < best_cst:
            print("⚠ Using text-based retrieval (distance: {:.4f})".format(best_text))
            retrieved_snippets = [code_snippets[idx][1] for idx in indices_text[0][:10] if idx < len(code_snippets)]
        else:
            print("✅ Using CST-based retrieval (distance: {:.4f})".format(best_cst))
            retrieved_snippets = [code_snippets[idx][1] for idx in indices_cst[0][:10] if idx < len(code_snippets)]

        context = "\n\n".join(retrieved_snippets)
        return None, context

    else:
        print("🌍 No relevant local match found. Falling back to generic AI.")
        return None, "No relevant code found."

def get_microsoft_response(question, context=None):
    """Generates a response using StarCoder-3B."""
    try:
        prompt = f"### Context:\n{context}\n\n### Question:\n{question}\n\n### Answer:"
        response = pipeline_model(prompt, max_length=300, do_sample=True, temperature=0.3)
        return response[0]["generated_text"].strip()
    except Exception as e:
        print("❌ StarCoder error:", traceback.format_exc())
        return f"Error: {str(e)}"

@app.route("/query", methods=["POST"])
def handle_query():
    try:
        data = request.json
        question = data.get("question", "")
        if not question:
            return jsonify({"error": "No question provided"}), 400

        print(f"📝 Received question: {question}")

        context = f"Project Overview from README:\n{readme_text}\n\n" if readme_text else ""
        result = retrieve_code_snippets(question)

        if result and result[0]:
            filename, snippet = result
            context += f"Additionally, here is relevant Python code:\n```python\n{snippet}\n```\n"
            print(f"📄 Found relevant code in: {filename}")

        return jsonify({"question": question, "answer": get_microsoft_response(question, context)})

    except Exception as e:
        print(f"❌ Server Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
