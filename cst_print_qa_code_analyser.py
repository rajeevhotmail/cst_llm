from flask import Flask, request, jsonify, render_template, Response
import os
import faiss
import pickle
import torch
import tree_sitter
import json
import time
from tree_sitter import Language, Parser
from sentence_transformers import SentenceTransformer
import requests
from requests.exceptions import ChunkedEncodingError
from dotenv import load_dotenv

# Configuration
PYTHON_FILES_DIR = "./repo"  # Directory containing Python files
BOLT_SERVER_URL = "http://107.21.146.84:8788"  # Cloned bolt.new server
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FAISS_CST_INDEX_FILE = "faiss_cst.idx"
FAISS_TEXT_INDEX_FILE = "faiss_text.idx"
CODE_SNIPPETS_FILE = "code_snippets.pkl"
README_FILE = "README.md"

# Initialize session with retry mechanism
session = requests.Session()
session.mount('http://', requests.adapters.HTTPAdapter(max_retries=3))

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
    cst_text = " ".join(cst_features)
    print("\n" + "="*50)
    print("🌳 CST Tree Structure:")
    print("="*50)
    print(cst_text[:200])
    print("="*50 + "\n")
    return cst_text

# Load FAISS and Code Snippets if They Exist
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

    THRESHOLD = 2.0  # Increased threshold for more matches
    filtered_indices_cst = [i for i, d in zip(indices_cst[0], distances_cst[0]) if d < THRESHOLD]
    filtered_indices_text = [i for i, d in zip(indices_text[0], distances_text[0]) if d < THRESHOLD]

    # Compare best distances from each method
    best_cst = distances_cst[0][0] if len(distances_cst[0]) > 0 else float('inf')
    best_text = distances_text[0][0] if len(distances_text[0]) > 0 else float('inf')

    if best_cst > THRESHOLD and best_text > THRESHOLD:
        print("🌍 No relevant local match found")
        return None, "No relevant code found."
    elif best_text < best_cst:
        print("⚠ Using text-based retrieval (distance: {:.4f})".format(best_text))
        return code_snippets[indices_text[0][0]]
    else:
        print("✅ Using CST-based retrieval (distance: {:.4f})".format(best_cst))
        return code_snippets[indices_cst[0][0]]

def handle_query_with_bolt(question, context=None):
    """Forwards query to bolt.new server with improved request handling"""
    print(f"🔄 Forwarding to bolt server: {BOLT_SERVER_URL}")

    try:
        # Prepare the messages array with context first if available
        messages = []
        if context:
            # Send context as part of the user's message in a structured format
            instruction = (
                "Context:\n"
                f"{context}\n"
                f"Question: {question}\n"
                "Please provide a structured analysis following this exact format:\n"
                "Based on the provided code and context, here's a structured analysis of the project:\n\n"
                "1. Purpose:\n"
                "   [Overview of the project's main goal]\n\n"
                "2. Main Components:\n"
                "   • [Component 1]\n"
                "   • [Component 2]\n\n"
                "3. Key Features:\n"
                "   • [Feature 1]\n"
                "   • [Feature 2]\n\n"
                "4. Technical Details:\n"
                "   • [Detail 1]\n"
                "   • [Detail 2]\n\n"
                "5. Additional Notes:\n"
                "   [Any other relevant observations]"
            )
            messages.append({
                "role": "user",
                "content": instruction
            })
        else:
            messages.append({
                "role": "user",
                "content": question
            })

        # Prepare request data
        data = {
            "messages": messages,
            "stream": True  # Enable streaming responses
        }

        endpoint = "/api/chat"
        url = f"{BOLT_SERVER_URL}{endpoint}"
        print(f"📡 Trying endpoint: {url}")
        print(f"📤 Sending data: {json.dumps(data, indent=2)}")

        # Browser-like headers
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream',  # For SSE support
            'Connection': 'keep-alive',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Origin': 'http://localhost:5000',  # Set to your server's origin
            'Referer': 'http://localhost:5000/',  # Set to your server's address
        }

        # Make request with streaming enabled
        with session.post(
            url,
            json=data,
            headers=headers,
            stream=True,
            timeout=30
        ) as response:
            print(f"Response status: {response.status_code}")

            if response.status_code == 200:
                print("✅ Connected to bolt server stream")
                full_response = []

                # Handle streaming response
                try:
                    for line in response.iter_lines():
                        if line:
                            line = line.decode('utf-8')
                            if line.startswith('0:"') and line.endswith('"'):
                                content = line[3:-1]  # Remove 0:" from start and " from end
                                content = content.replace('\\n', '\n')  # Convert \n to actual newlines
                                full_response.append(content)
                            else:
                                full_response.append(line)

                except ChunkedEncodingError as e:
                    print(f"⚠️ Chunked encoding error: {str(e)}")

                complete_response = ''.join(full_response)
                print(f"✅ Complete response assembled ({len(complete_response)} chars)")
                return complete_response
            else:
                error_msg = f"Bolt server error: {response.status_code}"
                print(f"❌ {error_msg}")
                print(f"Response content: {response.text}")
                return error_msg

    except requests.exceptions.RequestException as e:
        error_msg = f"Cannot connect to bolt server: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def handle_query():
    try:
        data = request.json
        question = data.get("question", "")
        if not question:
            return jsonify({"error": "No question provided"}), 400

        print(f"📝 Received question: {question}")

        # Start with README as base context if available
        context = f"Project Overview from README:\n{readme_text}\n\n" if readme_text else ""

        # Get relevant code context
        result = retrieve_code_snippets(question)
        if result and result[0]:
            filename, snippet = result
            context += f"""
Additionally, here is relevant Python code from file: {filename}
Code content:
```python
{snippet}
```
Please analyze this code and README to answer the question.
"""
            print(f"📄 Found relevant code in: {filename}")
        elif context:
            print("📄 Using README context only")
        else:
            print("⚠️ No context available")

        # Forward to bolt server and stream response
        answer = handle_query_with_bolt(question, context)
        return jsonify({"question": question, "answer": answer})

    except Exception as e:
        error_msg = f"Server error: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({"error": error_msg}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)