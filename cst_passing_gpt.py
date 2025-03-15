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
    """Retrieves the most relevant code snippet based on query."""
    if len(code_snippets) == 0:
        return None, "No code snippets available."

    query_embedding = embedding_model.encode(query, convert_to_numpy=True)
    distances, indices = faiss_index.search(query_embedding.reshape(1, -1), 3)

    THRESHOLD = 1.7
    filtered_indices = [i for i, d in zip(indices[0], distances[0]) if d < THRESHOLD]

    if not filtered_indices:
        return None, "No relevant snippet found."

    return code_snippets[filtered_indices[0]]


def query_chatgpt(question):
    """Queries OpenAI GPT-4o with retrieved code snippet."""
    result = retrieve_code_snippets(question)
    if result is None or result[0] is None:
        return "No relevant code snippet found."

    filename, snippet = result
    extracted_cst = extract_cst_features(snippet)

    prompt = f"""
    The following is a Python code snippet:
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


# Example Usage
if __name__ == "__main__":
    user_question = "What does the function main() do?"
    answer = query_chatgpt(user_question)
    print("ChatGPT Response:", answer)
