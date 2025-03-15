import openai
import os
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from tree_sitter import Language, Parser
import re
import logging

logging.basicConfig(level=logging.INFO)

# Configuration
BASE_DIR = os.getcwd()
PYTHON_FILES_DIR = os.path.join(BASE_DIR, "source_code")
FAISS_DB_DIR = os.path.join(BASE_DIR, "faiss_coarser_improved_db")
FAISS_CST_INDEX_FILE = os.path.join(FAISS_DB_DIR, "faiss_cst_functions.idx")
FAISS_TEXT_INDEX_FILE = os.path.join(FAISS_DB_DIR, "faiss_text_functions.idx")
CODE_SNIPPETS_FILE = os.path.join(FAISS_DB_DIR, "code_function_snippets.pkl")

# Ensure FAISS directory exists
os.makedirs(FAISS_DB_DIR, exist_ok=True)

# Locate README dynamically
README_FILE = None
for root, dirs, files in os.walk(PYTHON_FILES_DIR):
    if "README.md" in files:
        README_FILE = os.path.join(root, "README.md")
        break

# Load OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API Key. Set OPENAI_API_KEY environment variable.")

# Initialize Flask App
app = Flask(__name__)

# Load Sentence Transformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load Tree-sitter parser
PYTHON_LANGUAGE = Language("build/my-languages.so", "python")
parser = Parser()
parser.set_language(PY_LANGUAGE)

# FAISS Setup
dimension = 384
faiss_cst_index = faiss.IndexFlatL2(dimension)
faiss_text_index = faiss.IndexFlatL2(dimension)
code_snippets = []
readme_text = ""


# Extract function and class definitions from a Python file
def extract_functions_and_classes(filepath, source_code):
    """Extracts function and class definitions from Python code."""
    tree = parser.parse(source_code.encode("utf-8"))
    root_node = tree.root_node

    # Get the source code lines for line number references
    source_lines = source_code.splitlines()

    functions_and_classes = []

    def get_node_text(node):
        start_point, end_point = node.start_point, node.end_point
        if start_point[0] == end_point[0]:
            # Single line
            return source_lines[start_point[0]][start_point[1]:end_point[1]]
        else:
            # Multiple lines
            result = [source_lines[start_point[0]][start_point[1]:]]
            for i in range(start_point[0] + 1, end_point[0]):
                result.append(source_lines[i])
            result.append(source_lines[end_point[0]][:end_point[1]])
            return '\n'.join(result)

    def get_definition_type(node):
        if node.type == "function_definition":
            # Find the function name node
            for child in node.children:
                if child.type == "identifier":
                    return "function", child.text.decode("utf-8")
        elif node.type == "class_definition":
            # Find the class name node
            for child in node.children:
                if child.type == "identifier":
                    return "class", child.text.decode("utf-8")
        return None, None

    def traverse_for_definitions(node):
        if node.type in ["function_definition", "class_definition"]:
            def_type, name = get_definition_type(node)
            if def_type and name:
                code_text = get_node_text(node)
                start_line = node.start_point[0] + 1  # 1-indexed line numbers
                end_line = node.end_point[0] + 1

                comments, docstring = extract_comments_and_docstrings(code_text)

                functions_and_classes.append({
                    "type": def_type,
                    "name": name,
                    "code": code_text,
                    "start_line": start_line,
                    "end_line": end_line,
                    "filepath": filepath,
                    "comments": comments,
                    "docstring": docstring
                })
                logging.info(f"Extracted {def_type}: {name} from {filepath} (lines {start_line}-{end_line})")
                # If it's a class, also extract its methods
                if def_type == "class":
                    for child in node.children:
                        if child.type == "block":
                            for method_node in child.children:
                                if method_node.type == "function_definition":
                                    method_type, method_name = get_definition_type(method_node)
                                    if method_type and method_name:
                                        method_code = get_node_text(method_node)
                                        method_start = method_node.start_point[0] + 1
                                        method_end = method_node.end_point[0] + 1

                                        method_comments, method_docstring = extract_comments_and_docstrings(method_code)

                                        functions_and_classes.append({
                                            "type": "method",
                                            "name": method_name,
                                            "class_name": name,
                                            "code": method_code,
                                            "start_line": method_start,
                                            "end_line": method_end,
                                            "filepath": filepath,
                                            "comments": method_comments,
                                            "docstring": method_docstring
                                        })
                                        logging.info(f"Extracted method: {method_name} from class {name} (lines {method_start}-{method_end})")
        for child in node.children:
            traverse_for_definitions(child)

    traverse_for_definitions(root_node)
    return functions_and_classes

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

# Get a descriptive title for the code snippet
def get_snippet_title(snippet):
    if snippet["type"] == "function":
        logging.info(f"Extracted function: {snippet['name']}")
        return f"Function: {snippet['name']}"
    elif snippet["type"] == "class":
        logging.info(f"Extracted class: {snippet['name']}")
        return f"Class: {snippet['name']}"
    elif snippet["type"] == "method":
        logging.info(f"Extracted method: {snippet['class_name']}.{snippet['name']}")
        return f"Method: {snippet['class_name']}.{snippet['name']}"
    return "Code Snippet"

# Extract comments and docstrings from the code snippet
def extract_comments_and_docstrings(code):
    """
    Extract comments and docstrings from Python code more thoroughly.

    Args:
        code (str): Python code to analyze

    Returns:
        tuple: (comments list, docstring)
    """
    # Extract inline comments
    comments = re.findall(r'#.*', code)

    # Extract docstrings - look for triple quotes more thoroughly
    # This handles both """ and ''' style docstrings
    docstring_match = re.search(r'(?:"""|\'\'\')(.*?)(?:"""|\'\'\')|\'\'\'\s*(.*?)\s*\'\'\'', code, re.DOTALL)

    if docstring_match:
        # Get the captured group - could be in first or second position
        docstring = docstring_match.group(1) if docstring_match.group(1) else docstring_match.group(2)
        docstring = docstring.strip() if docstring else ""
    else:
        docstring = ""

    # If we have no docstring but have function/class-level comments right before definition,
    # use those as a substitute
    if not docstring and comments:
        # Look for comment blocks right before the definition
        lines = code.split('\n')
        definition_line = -1

        for i, line in enumerate(lines):
            if re.search(r'^\s*def\s+\w+|^\s*class\s+\w+', line):
                definition_line = i
                break

        if definition_line > 0:
            # Check for comment block above definition
            comment_block = []
            i = definition_line - 1

            while i >= 0 and (lines[i].strip().startswith('#') or not lines[i].strip()):
                if lines[i].strip().startswith('#'):
                    comment_block.insert(0, lines[i].strip()[1:].strip())  # Remove # and whitespace
                i -= 1

            if comment_block:
                docstring = "\n".join(comment_block)

    return comments, docstring

# Check and initialize FAISS indices if missing
if os.path.exists(FAISS_CST_INDEX_FILE) and os.path.exists(FAISS_TEXT_INDEX_FILE) and os.path.exists(CODE_SNIPPETS_FILE):
    print("✅ Loading precomputed FAISS CST and text indexes...")
    faiss_cst_index = faiss.read_index(FAISS_CST_INDEX_FILE)
    faiss_text_index = faiss.read_index(FAISS_TEXT_INDEX_FILE)
    with open(CODE_SNIPPETS_FILE, "rb") as f:
        code_snippets = pickle.load(f)
    print(f"✅ Loaded {len(code_snippets)} code snippets")
else:
    print("🔄 No FAISS indexes found. Initializing from scratch...")
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

                            # Extract comments and docstrings
                            comments, docstring = extract_comments_and_docstrings(code)

                            # Create text embedding
                            text_embedding = embedding_model.encode(code, convert_to_numpy=True)
                            logging.info(f"Adding to FAISS text index: {item['type']} - {item['name']} from {item['filepath']}")
                            faiss_text_index.add(text_embedding.reshape(1, -1))

                            # Create CST embedding
                            cst_text = extract_cst_features(code)
                            cst_embedding = embedding_model.encode(cst_text, convert_to_numpy=True)
                            logging.info(f"Adding to FAISS CST index: {item['type']} - {item['name']} from {item['filepath']}")
                            faiss_cst_index.add(cst_embedding.reshape(1, -1))

                            # Add to code snippets
                            item["comments"] = comments
                            item["docstring"] = docstring
                            code_snippets.append(item)
                except Exception as e:
                    print(f"❌ Error processing {filepath}: {str(e)}")

    print(f"✅ Indexed {len(code_snippets)} functions and classes")

    faiss.write_index(faiss_text_index, FAISS_TEXT_INDEX_FILE)
    faiss.write_index(faiss_cst_index, FAISS_CST_INDEX_FILE)
    with open(CODE_SNIPPETS_FILE, "wb") as f:
        pickle.dump(code_snippets, f)

# Load README if available
if README_FILE and os.path.exists(README_FILE):
    with open(README_FILE, "r", encoding="utf-8") as f:
        readme_text = f.read()
        print("✅ README.md loaded for project summary!")

def retrieve_code_snippets(query, top_k=20):  # Increased from 15 to 20
    """Retrieves the most relevant code snippets based on query."""
    if len(code_snippets) == 0:
        return None, "No code snippets available."

    query_embedding = embedding_model.encode(query, convert_to_numpy=True)
    distances_cst, indices_cst = faiss_cst_index.search(query_embedding.reshape(1, -1), top_k)
    distances_text, indices_text = faiss_text_index.search(query_embedding.reshape(1, -1), top_k)

    print(f"🔍 Query: {query}")
    print(f"📏 CST Distances: {distances_cst}")
    print(f"📏 Text Distances: {distances_text}")

    THRESHOLD = 3.0  # Increase threshold to include more results
    best_cst = distances_cst[0][0] if len(distances_cst[0]) > 0 else float('inf')
    best_text = distances_text[0][0] if len(distances_text[0]) > 0 else float('inf')

    # Prepare to store snippets by type so we can balance them
    class_snippets = []
    function_snippets = []
    method_snippets = []

    # Special case for "what are functions" or similar queries
    if "function" in query.lower() or "class" in query.lower():
        # Use all indices from both retrievals to get a comprehensive picture
        all_indices = set(list(indices_cst[0]) + list(indices_text[0]))

        # Filter out invalid indices
        valid_indices = [idx for idx in all_indices if idx < len(code_snippets)]

        # Categorize snippets
        for idx in valid_indices:
            snippet = code_snippets[idx]
            if snippet["type"] == "function":
                function_snippets.append(snippet)
            elif snippet["type"] == "class":
                class_snippets.append(snippet)
            elif snippet["type"] == "method":
                method_snippets.append(snippet)
    else:
        # Standard case - use best retrieval method
        if best_text < best_cst:
            print("⚠ Using text-based retrieval (distance: {:.4f})".format(best_text))
            print("⚠ FORCING CST retrieval (distance: {:.4f})".format(best_text))
            #retrieved_indices = indices_text[0][:top_k]
            retrieved_indices = indices_cst[0][:top_k]
        else:
            print("✅ Using CST-based retrieval (distance: {:.4f})".format(best_cst))
            retrieved_indices = indices_cst[0][:top_k]

        # Filter out invalid indices
        retrieved_indices = [idx for idx in retrieved_indices if idx < len(code_snippets)]

        # Categorize snippets
        for idx in retrieved_indices:
            snippet = code_snippets[idx]
            if snippet["type"] == "function":
                function_snippets.append(snippet)
            elif snippet["type"] == "class":
                class_snippets.append(snippet)
            elif snippet["type"] == "method":
                method_snippets.append(snippet)

    # Ensure we have a balance of types, prioritizing what's asked for
    retrieved_items = []

    # If query mentions functions, prioritize them
    if "function" in query.lower():
        # Include all functions, but limit other types
        retrieved_items.extend(function_snippets[:min(15, len(function_snippets))])
        retrieved_items.extend(class_snippets[:min(5, len(class_snippets))])
        retrieved_items.extend(method_snippets[:min(5, len(method_snippets))])
    # If query mentions classes, prioritize them
    elif "class" in query.lower():
        # Include all classes, but limit other types
        retrieved_items.extend(class_snippets[:min(15, len(class_snippets))])
        retrieved_items.extend(function_snippets[:min(5, len(function_snippets))])
        retrieved_items.extend(method_snippets[:min(5, len(method_snippets))])
    # Otherwise balance
    else:
        # Balance between types
        retrieved_items.extend(function_snippets[:min(7, len(function_snippets))])
        retrieved_items.extend(class_snippets[:min(7, len(class_snippets))])
        retrieved_items.extend(method_snippets[:min(6, len(method_snippets))])

    # Special case - if we found ZERO functions but query asks about functions
    if len(function_snippets) == 0 and "function" in query.lower():
        print("⚠️ No functions found through embedding search. Doing direct search...")
        # Do a direct search for functions
        function_indices = []
        for i, snippet in enumerate(code_snippets):
            if snippet["type"] == "function":
                function_indices.append(i)

        # Add some random functions if we found any
        import random
        if function_indices:
            random_functions = random.sample(
                function_indices,
                min(10, len(function_indices))
            )
            for idx in random_functions:
                retrieved_items.append(code_snippets[idx])

    print(f"INFO:root:Retrieved {len(function_snippets)} functions and {len(class_snippets)} classes")

    if not retrieved_items:
        print("🌍 No relevant local match found. Falling back to generic AI.")
        return None, "No relevant code found."

    # Format the retrieved snippets into a context string
    context_parts = []
    for snippet in retrieved_items:
        snippet_type = snippet["type"]
        name = snippet["name"]
        code = snippet["code"]
        filepath = snippet["filepath"]
        comments = "\n".join(snippet["comments"]) if snippet["comments"] else "No comments available"
        docstring = snippet["docstring"] if snippet["docstring"] else "No docstring available"

        if snippet_type == "method":
            class_name = snippet["class_name"]
            header = f"Method: {class_name}.{name} from {filepath} (lines {snippet['start_line']}-{snippet['end_line']})"
        else:
            header = f"{snippet_type.capitalize()}: {name} from {filepath} (lines {snippet['start_line']}-{snippet['end_line']})"

        context_parts.append(f"{header}\n```python\n{code}\n```\n\nComments:\n{comments}\n\nDocstring:\n{docstring}\n")

    return None, "\n".join(context_parts)
    # Format the retrieved snippets into a context string
    context_parts = []
    for snippet in retrieved_items:
        snippet_type = snippet["type"]
        name = snippet["name"]
        code = snippet["code"]
        filepath = snippet["filepath"]
        comments = "\n".join(snippet["comments"]) if snippet["comments"] else "No comments available"
        docstring = snippet["docstring"] if snippet["docstring"] else "No docstring available"

        if snippet_type == "method":
            class_name = snippet["class_name"]
            header = f"Method: {class_name}.{name} from {filepath} (lines {snippet['start_line']}-{snippet['end_line']})"
        else:
            header = f"{snippet_type.capitalize()}: {name} from {filepath} (lines {snippet['start_line']}-{snippet['end_line']})"

        context_parts.append(f"{header}\n```python\n{code}\n```\n\nComments:\n{comments}\n\nDocstring:\n{docstring}\n")

    return None, "\n".join(context_parts)

def get_openai_response(question, context=None):
    try:
        # Updated system prompt with Markdown formatting instructions
        system_prompt = (
            "You are an AI assistant for analyzing a Python project. "
            "Your goal is to provide detailed insights into the project's structure using the provided context. "
            "The context contains functions, classes, and methods from the codebase, along with associated comments and docstrings. "
            "\n\n"
            "IMPORTANT INSTRUCTIONS:\n"
            "1. You MUST list ALL functions and classes that appear in the context.\n"
            "2. If the context has functions, list them under the '## Functions' heading.\n"
            "3. If the context has classes, list them under the '## Classes' heading.\n"
            "4. If NO functions are found in the context, explicitly state 'I found no standalone functions in the provided context.'\n"
            "5. Similarly, if NO classes are found, state 'I found no classes in the provided context.'\n"
            "\n"
            "For each item in your response:\n"
            "- Bold the name using markdown (**Name**)\n"
            "- Show location (file path and line numbers)\n"
            "- Include a brief description based on available comments or docstrings\n"
            "- If no description is available, say 'No description available'\n"
            "\n"
            "FORMATTING:\n"
            "- Format your ENTIRE response as clean Markdown (.md) format\n"
            "- Use proper Markdown headings (##), numbered lists (1., 2., etc.), and bold text (**text**)\n"
            "- For code examples, use Markdown code blocks with Python syntax highlighting (```python)\n"
            "- Make the response clean, well-structured, and easy to read as a Markdown document\n"
            "\n"
            "Remember to include ALL functions and classes from the context, organized in clear sections."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"}
        ]

        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        import traceback
        print("❌ OpenAI API call failed:", traceback.format_exc())
        return f"OpenAI API error: {str(e)}"

def filter_irrelevant_info(response, context):
    # Split the response into sentences
    sentences = response.split(". ")

    relevant_sentences = []
    for sentence in sentences:
        # Check if the sentence contains any information from the context
        if any(snippet in sentence for snippet in context.split("\n")):
            relevant_sentences.append(sentence)

    # Join the relevant sentences back into a coherent response
    filtered_response = ". ".join(relevant_sentences)

    return filtered_response

def debug_context_and_prompt(question, context, debug_mode=True):
    """
    Provides debug information about the context and prompt before sending to OpenAI API.
    In console mode, asks whether to proceed. In web mode, just logs info.

    Args:
        question (str): User's question
        context (str): Context retrieved from code snippets
        debug_mode (bool): Whether to enable debug mode

    Returns:
        bool: True if should proceed with API call, False otherwise
    """
    if not debug_mode:
        return True

    print("\n" + "="*80)
    print("🔍 DEBUG: CONTEXT AND PROMPT INFORMATION")
    print("="*80)

    # 1. Basic statistics
    context_length = len(context) if context else 0
    token_estimate = context_length / 4  # Rough estimate: ~4 chars per token

    print(f"📊 Question length: {len(question)} chars")
    print(f"📊 Context length: {context_length} chars (approx. {token_estimate:.0f} tokens)")

    # 2. Snippet analysis
    if context:
        function_count = context.count("Function: ")
        class_count = context.count("Class: ")
        method_count = context.count("Method: ")

        print(f"📚 Content summary:")
        print(f"   - Functions: {function_count}")
        print(f"   - Classes: {class_count}")
        print(f"   - Methods: {method_count}")

        # Check if context has at least some functions and classes
        if function_count == 0 and "what are functions" in question.lower():
            print("⚠️ WARNING: No functions found in context but question asks about functions!")

            # Only ask for confirmation in terminal mode - not in web mode
            if "__TERMINAL_MODE__" in globals() and __TERMINAL_MODE__:
                proceed = input("Proceed with API call? (y/n): ").lower().strip() == 'y'
                return proceed
            # In web mode, just warn but proceed
            return True
    else:
        print("⚠️ No context available!")

        # Only ask for confirmation in terminal mode - not in web mode
        if "__TERMINAL_MODE__" in globals() and __TERMINAL_MODE__:
            proceed = input("Proceed with empty context? (y/n): ").lower().strip() == 'y'
            return proceed
        # In web mode, just warn but proceed
        return True

    # 3. Preview
    print("\n📝 Context preview (first 500 chars):")
    print("-"*80)
    print(context[:500] + "..." if len(context) > 500 else context)
    print("-"*80)

    # 4. Cost estimate
    cost_estimate = (token_estimate + 1000) / 1000 * 0.03  # Very rough cost estimate for GPT-4
    print(f"💰 Estimated API cost: ${cost_estimate:.4f} USD")

    # NEW CODE: Always pause before sending to API
    print("\n🛑 Ready to send to API. Press Enter to continue or type 'skip' to abort: ")
    user_input = input().strip().lower()

    if user_input == "skip":
        print("✋ API call aborted by user")
        return False

    print("✅ Proceeding with API call...")
    return True

# Update your route handler to use this
@app.route("/query", methods=["POST"])
def handle_query():
    try:
        data = request.json
        question = data.get("question", "")
        debug_mode = data.get("debug_mode", True)  # Get debug mode from request

        if not question:
            return jsonify({"error": "No question provided"}), 400
        print(f"📝 Received question: {question}")

        # Create a base context with the README information
        context = f"Project Overview from README:\n{readme_text}\n\n" if readme_text else ""

        # Get relevant code snippets
        _, code_context = retrieve_code_snippets(question)
        if code_context != "No relevant code found." and code_context != "No code snippets available.":
            context += f"Relevant Code Snippets:\n\n{code_context}"
            print(f"📄 Found relevant code snippets")
        elif context:
            print("📄 Using README context only")
        else:
            print("⚠️ No context available")

        # Add debugging step
        should_proceed = debug_context_and_prompt(question, context, debug_mode)

        if not should_proceed:
            # API call was aborted by user
            return jsonify({
                "question": question,
                "answer": "## API Call Aborted\n\nThe API call was manually aborted. Check server logs for context details."
            })

        response = get_openai_response(question, context)
        filtered_response = filter_irrelevant_info(response, context)

        return jsonify({"question": question, "answer": filtered_response})

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
