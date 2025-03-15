import os
import faiss
import torch
import tree_sitter
from tree_sitter import Language, Parser
from sentence_transformers import SentenceTransformer
import openai
from dotenv import load_dotenv

# Load the .env file
load_dotenv("openai_key.env")

# Get the API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Verify if the key is loaded
if openai.api_key:
    print("✅ OpenAI API key loaded successfully!")
else:
    print("❌ OpenAI API key is missing. Check your openai_key.env file.")

# Configuration
PYTHON_FILES_DIR = "./repo"  # Directory containing Python files
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Hugging Face sentence transformer
FAISS_INDEX_FILE = "faiss_index.idx"

# Load Sentence Transformer model
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Load Tree-sitter Python parser
PYTHON_LANGUAGE = Language("build/my-languages.so", "python")
parser = Parser()
parser.set_language(PYTHON_LANGUAGE)

# FAISS Setup
dimension = 384  # Embedding dimension for MiniLM-L6-v2
faiss_index = faiss.IndexFlatL2(dimension)

# Store code snippets and embeddings
code_snippets = []
stored_embeddings = []

def extract_cst_features(source_code):
    """Extracts a simplified CST representation from Python code."""
    tree = parser.parse(source_code.encode("utf-8"))
    root_node = tree.root_node
    cst_features = []

    def traverse(node):
        if node.child_count == 0:
            return
        cst_features.append(node.type)
        for child in node.children:
            traverse(child)

    traverse(root_node)
    return " ".join(cst_features)

# Check if the directory exists
if not os.path.exists(PYTHON_FILES_DIR):
    print(f"❌ Error: The directory '{PYTHON_FILES_DIR}' does not exist!")
else:
    print(f"✅ Directory '{PYTHON_FILES_DIR}' exists.")
    print("Scanning for Python files...")

    # Step 1: Parse Python Files and Generate Embeddings
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
                    stored_embeddings.append(embedding)
                    code_snippets.append((filepath, source_code))

print(f"Total Python files processed: {len(code_snippets)}")

# Save FAISS index
faiss.write_index(faiss_index, FAISS_INDEX_FILE)
print(f"FAISS Index Size: {faiss_index.ntotal}")

def retrieve_code_snippets(query):
    """Retrieves the most relevant code snippet from FAISS index based on query."""
    if len(code_snippets) == 0:
        print("⚠ No code snippets available!")
        return None, "No code snippets available."

    query_embedding = embedding_model.encode(query, convert_to_numpy=True)
    distances, indices = faiss_index.search(query_embedding.reshape(1, -1), 1)

    print(f"Query: {query}")
    print(f"Distances: {distances}")
    print(f"Indices: {indices}")

    if indices[0][0] == -1 or indices[0][0] >= len(code_snippets):
        print("⚠ No relevant match found in FAISS.")
        return None, "No relevant snippet found."

    return code_snippets[indices[0][0]]

def query_chatgpt(question):
    """Queries OpenAI GPT-4o with retrieved code snippet."""
    result = retrieve_code_snippets(question)
    if result is None or result[0] is None:
        return "No relevant code snippet found."

    filename, snippet = result
    prompt = f"""
    The following is a Python code snippet:
    ```python
    {snippet}
    ```
    Answer the user's question based on the above code.
    Question: {question}
    """
    client = openai.OpenAI()  # Initialize the client

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI that explains Python code."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

# Example Usage
if __name__ == "__main__":
    user_question = "What does the function main() do?"
    answer = query_chatgpt(user_question)
    print("ChatGPT Response:", answer)
