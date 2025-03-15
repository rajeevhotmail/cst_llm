import os
import re
import faiss
import pickle
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
from tree_sitter import Language, Parser

# Configuration
BASE_DIR = os.getcwd()
PYTHON_FILES_DIR = os.path.join(BASE_DIR, "source_code")
FAISS_DB_DIR = os.path.join(BASE_DIR, "faiss_cody_db")
FAISS_CST_INDEX_FILE = os.path.join(FAISS_DB_DIR, "faiss_cst_functions.idx")
FAISS_TEXT_INDEX_FILE = os.path.join(FAISS_DB_DIR, "faiss_text_functions.idx")
CODE_SNIPPETS_FILE = os.path.join(FAISS_DB_DIR, "code_function_snippets.pkl")
# Ensure FAISS directory exists
os.makedirs(FAISS_DB_DIR, exist_ok=True)
MAX_TOKENS_PER_SNIPPET = 1000
TOKEN_BUDGET = 6000
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

README_FILE = None
for root, dirs, files in os.walk(PYTHON_FILES_DIR):
    if "README.md" in files:
        README_FILE = os.path.join(root, "README.md")
        break

# Load OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API Key. Set OPENAI_API_KEY environment variable.")

# Setup logging
logging.basicConfig(level=logging.INFO)

# Initialize embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Initialize tree-sitter parser
Language.build_library('build/my-languages.so', ['vendor/tree-sitter-python'])
PY_LANGUAGE = Language('build/my-languages.so', 'python')
parser = Parser()
parser.set_language(PY_LANGUAGE)

# Initialize FAISS indices
dimension = embedding_model.get_sentence_embedding_dimension()
faiss_cst_index = faiss.IndexFlatL2(dimension)
faiss_text_index = faiss.IndexFlatL2(dimension)
code_snippets = []

def is_overview_question(query):
    """Determine if a query is asking for a project overview."""
    overview_patterns = [
        "what is", "what's", "what are", "what does",
        "overview", "about", "purpose", "goal", "aim",
        "describe", "summary", "summarize"
    ]
    query_lower = query.lower()
    return any(pattern in query_lower for pattern in overview_patterns)

def get_file_importance_score(filepath, is_overview=False):
    """Score the importance of a file for answering queries."""
    filename = os.path.basename(filepath).lower()

    # High importance files for overview questions
    if is_overview:
        if "readme" in filename or "documentation" in filename:
            return 1.0
        if filename == "__init__.py":
            return 0.9
        if filename in ["main.py", "app.py", "api.py"]:
            return 0.8
        if "setup.py" in filename or "pyproject.toml" in filename:
            return 0.7
        # Deprioritize tests and utility scripts for overviews
        if "test" in filepath or "script" in filepath:
            return 0.2

    # General importance scoring
    if "test" in filepath:
        return 0.4
    if "example" in filepath:
        return 0.6
    if "util" in filepath:
        return 0.5

    # Default score
    return 0.5

# PART 1: CODE EXTRACTION FUNCTIONS
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

# PART 2: CODE CHUNKING AND SUMMARIZATION
def chunk_large_code_unit(snippet, max_lines=50, overlap=10):
    """Split large code units into smaller chunks with some overlap."""
    if snippet["end_line"] - snippet["start_line"] <= max_lines:
        return [snippet]

    code_lines = snippet["code"].split('\n')
    chunks = []

    # Extract the signature/header separately (for classes/functions)
    header_lines = []
    body_start = 0

    if snippet["type"] in ["class", "function", "method"]:
        # Find where the body starts (after the signature)
        for i, line in enumerate(code_lines):
            header_lines.append(line)
            if line.strip().endswith(':'):
                body_start = i + 1
                break

    # Create chunks with the header + portions of the body
    for i in range(body_start, len(code_lines), max_lines - overlap - len(header_lines)):
        end_idx = min(i + max_lines - len(header_lines), len(code_lines))

        # Create a new snippet for this chunk
        chunk = snippet.copy()
        chunk["code"] = '\n'.join(header_lines + code_lines[i:end_idx])
        chunk["start_line"] = snippet["start_line"] + i
        chunk["end_line"] = snippet["start_line"] + end_idx - 1
        chunk["is_chunk"] = True
        chunk["chunk_info"] = f"Part {(i-body_start)//(max_lines-overlap-len(header_lines))+1}"

        chunks.append(chunk)

    return chunks

def summarize_large_code_unit(snippet):
    """Create a summarized version of a large code unit."""
    code_lines = snippet["code"].split('\n')

    # Always keep the first few lines (signature/class definition)
    summary_lines = []
    in_docstring = False
    docstring_delimiter = None

    for i, line in enumerate(code_lines):
        # Always keep the first few lines (definition)
        if i < 5:
            summary_lines.append(line)

            # Check for docstring start
            if '"""' in line or "'''" in line:
                in_docstring = True
                docstring_delimiter = '"""' if '"""' in line else "'''"
            continue

        # Keep docstring content
        if in_docstring:
            summary_lines.append(line)
            if docstring_delimiter in line:
                in_docstring = False
            continue

        # For classes, keep method definitions but summarize bodies
        if snippet["type"] == "class" and (line.strip().startswith('def ') or
                                          line.strip().startswith('async def ')):
            summary_lines.append(line)
            # Add a placeholder for method body
            if line.strip().endswith(':'):
                summary_lines.append("    # Method implementation (summarized)")
            continue

        # Keep important structural elements like property definitions
        if line.strip().startswith('@property') or line.strip().startswith('class '):
            summary_lines.append(line)
            continue

    # Create a summarized snippet
    summarized = snippet.copy()
    summarized["code"] = '\n'.join(summary_lines)
    summarized["is_summarized"] = True

    return summarized

def extract_code_metadata(snippet):
    """Extract just the signature, docstring and structure without implementation details."""
    metadata = {
        "type": snippet["type"],
        "name": snippet["name"],
        "filepath": snippet["filepath"],
        "start_line": snippet["start_line"],
        "end_line": snippet["end_line"],
        "docstring": snippet["docstring"],
    }

    # Extract just the signature for functions/methods
    if snippet["type"] in ["function", "method"]:
        code_lines = snippet["code"].split('\n')
        for i, line in enumerate(code_lines):
            if line.strip().endswith(':'):
                metadata["signature"] = '\n'.join(code_lines[:i+1])
                # Also set the code key to the signature
                metadata["code"] = metadata["signature"]
                break

    # For classes, extract class definition and method signatures
    elif snippet["type"] == "class":
        signatures = []
        code_lines = snippet["code"].split('\n')

        # Get class definition
        for i, line in enumerate(code_lines):
            if line.strip().startswith('class ') and line.strip().endswith(':'):
                signatures.append(line)
                break

        # Get method signatures
        in_method_def = False
        method_def_lines = []

        for line in code_lines:
            stripped = line.strip()
            if stripped.startswith('def ') or stripped.startswith('async def '):
                in_method_def = True
                method_def_lines = [line]
            elif in_method_def:
                method_def_lines.append(line)
                if stripped.endswith(':'):
                    signatures.append('\n'.join(method_def_lines))
                    in_method_def = False

        metadata["signatures"] = signatures
        # Set the code key to all signatures joined
        metadata["code"] = '\n'.join(signatures)

    # Ensure there's always a code key
    if "code" not in metadata:
        if "signature" in metadata:
            metadata["code"] = metadata["signature"]
        elif "signatures" in metadata:
            metadata["code"] = '\n'.join(metadata["signatures"])
        else:
            metadata["code"] = f"# {metadata['type']}: {metadata['name']}"

    return metadata

def get_mock_response(question, context):
    return f"""
## Code Analysis Results

I analyzed the code related to your question: "{question}"

Here's what I found in the codebase:

{context}

This is a simplified response for testing purposes. In production, you would use an LLM to generate more detailed analysis.
"""

# PART 3: RERANKING FUNCTIONS
def rerank_snippets(query, retrieved_snippets, max_to_keep=10):
    """Rerank retrieved snippets before sending to LLM with error handling."""
    if not retrieved_snippets:
        print("⚠️ Warning: No snippets to rerank")
        return []

    # Print debug info about the first snippet
    if retrieved_snippets:
        print(f"🔍 First snippet keys: {list(retrieved_snippets[0].keys())}")

    # Check if this is an overview question
    is_overview = is_overview_question(query)

    # Calculate more precise relevance scores
    scored_snippets = []
    for snippet in retrieved_snippets:
        try:
            # Calculate a combined score from multiple factors
            exact_match_score = calculate_keyword_match(query, snippet)
            doc_quality_score = score_documentation_quality(snippet)

            # Use initial_score if available, otherwise default to 0.5
            initial_score = snippet.get("initial_score", 0.5)

            # Use importance_score if available
            importance_score = snippet.get("importance_score", 0.5)

            # Adjust weights based on query type
            if is_overview:
                # For overview questions, prioritize documentation and important files
                final_score = 0.3 * exact_match_score + 0.4 * doc_quality_score + 0.1 * initial_score + 0.2 * importance_score
            else:
                # For specific questions, prioritize exact matches
                final_score = 0.5 * exact_match_score + 0.2 * doc_quality_score + 0.2 * initial_score + 0.1 * importance_score

            scored_snippets.append((snippet, final_score))
        except Exception as e:
            print(f"❌ Error scoring snippet: {str(e)}")
            print(f"Problematic snippet: {snippet}")

    if not scored_snippets:
        print("⚠️ Warning: No snippets were successfully scored")
        return []

    # Sort by score
    scored_snippets.sort(key=lambda x: x[1], reverse=True)

    # Apply diversity filtering (optional)
    try:
        diverse_snippets = apply_diversity_filter(scored_snippets)
    except Exception as e:
        print(f"❌ Error in diversity filtering: {str(e)}")
        diverse_snippets = scored_snippets

    # Return top K snippets
    return [s[0] for s in diverse_snippets[:max_to_keep]]

def calculate_keyword_match(query, snippet):
    """Calculate how well the snippet matches query keywords."""
    # Extract keywords from query
    query_words = set(query.lower().split())

    # Get snippet text
    snippet_text = ""
    if "name" in snippet:
        snippet_text += snippet["name"].lower() + " "
    if "code" in snippet:
        snippet_text += snippet["code"].lower() + " "
    if "docstring" in snippet:
        snippet_text += snippet["docstring"].lower() + " "

    # Count matches
    matches = 0
    for word in query_words:
        if word in snippet_text:
            matches += 1

    # Calculate score
    if len(query_words) > 0:
        return matches / len(query_words)
    else:
        return 0.0

def score_documentation_quality(snippet):
    """Score the quality of documentation in a snippet."""
    score = 0.5  # Default score

    # Check for docstring
    docstring = snippet.get("docstring", "")
    if docstring and docstring != "No docstring available":
        # Longer docstrings are usually more informative
        score += min(len(docstring) / 500, 0.3)  # Up to 0.3 for length

        # Check for specific documentation patterns
        if "param" in docstring or "parameter" in docstring:
            score += 0.1  # Documents parameters
        if "return" in docstring:
            score += 0.1  # Documents return values
        if "example" in docstring.lower():
            score += 0.1  # Has examples

    # Check for comments
    comments = snippet.get("comments", [])
    if comments and len(comments) > 0:
        score += min(len(comments) / 10, 0.2)  # Up to 0.2 for comments

    return min(score, 1.0)  # Cap at 1.0
def apply_diversity_filter(scored_snippets, diversity_threshold=0.3):
    """Apply diversity filtering to avoid redundant snippets."""
    if not scored_snippets:
        return []

    # Start with the highest scored snippet
    filtered_snippets = [scored_snippets[0]]

    # Consider remaining snippets
    for candidate in scored_snippets[1:]:
        # Check if this candidate is too similar to any already selected snippet
        is_diverse = True
        for selected in filtered_snippets:
            similarity = calculate_snippet_similarity(candidate[0], selected[0])
            if similarity > diversity_threshold:
                is_diverse = False
                break

        # If it's diverse enough, add it
        if is_diverse:
            filtered_snippets.append(candidate)

    return filtered_snippets

def calculate_snippet_similarity(snippet1, snippet2):
    """Calculate similarity between two code snippets."""
    # Simple implementation: check if they're from the same file
    if snippet1.get("filepath") == snippet2.get("filepath"):
        # If they're from the same file, check if they overlap
        s1_start = snippet1.get("start_line", 0)
        s1_end = snippet1.get("end_line", 0)
        s2_start = snippet2.get("start_line", 0)
        s2_end = snippet2.get("end_line", 0)

        # Check for line overlap
        if (s1_start <= s2_end and s2_start <= s1_end):
            return 0.8  # High similarity for overlapping code
        else:
            return 0.5  # Medium similarity for same file

    # Check if they have the same name (could be overloaded functions)
    if snippet1.get("name") == snippet2.get("name"):
        return 0.4

    # Otherwise, they're probably different
    return 0.1

def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def select_best_chunk(query, chunks):
    """Select the most relevant chunk for a query."""
    query_embedding = embedding_model.encode(query, convert_to_numpy=True)

    best_score = float('-inf')
    best_chunk = chunks[0]  # Default to first chunk

    for chunk in chunks:
        chunk_text = chunk["code"]
        chunk_embedding = embedding_model.encode(chunk_text, convert_to_numpy=True)

        # Calculate cosine similarity
        similarity = np.dot(query_embedding, chunk_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
        )

        if similarity > best_score:
            best_score = similarity
            best_chunk = chunk

    return best_chunk

# PART 4: MAIN RETRIEVAL FUNCTION
def retrieve_code_snippets(query, top_k=20):
    """Retrieves the most relevant code snippets based on query."""
    if len(code_snippets) == 0:
        return None, "No code snippets available."

    # Get initial candidates from both indices
    query_embedding = embedding_model.encode(query, convert_to_numpy=True)
    distances_cst, indices_cst = faiss_cst_index.search(query_embedding.reshape(1, -1), top_k)
    distances_text, indices_text = faiss_text_index.search(query_embedding.reshape(1, -1), top_k)

    print(f"🔍 Query: {query}")
    print(f"📏 CST Distances: {distances_cst}")
    print(f"📏 Text Distances: {distances_text}")

    # Check if this is an overview question
    is_overview = is_overview_question(query)

    # Determine which retrieval method to use
    best_cst = distances_cst[0][0] if len(distances_cst[0]) > 0 else float('inf')
    best_text = distances_text[0][0] if len(distances_text[0]) > 0 else float('inf')

    # For natural language queries, prefer text embeddings
    if "what" in query.lower() or "how" in query.lower() or "why" in query.lower():
        print("⚠ Using text-based retrieval for natural language query")
        retrieved_indices = indices_text[0][:top_k]
    else:
        # Otherwise use the better performing index
        if best_text < best_cst:
            print("⚠ Using text-based retrieval (distance: {:.4f})".format(best_text))
            retrieved_indices = indices_text[0][:top_k]
        else:
            print("✅ Using CST-based retrieval (distance: {:.4f})".format(best_cst))
            retrieved_indices = indices_cst[0][:top_k]

    # Filter out invalid indices
    retrieved_indices = [idx for idx in retrieved_indices if idx < len(code_snippets)]

    # If it's an overview question, prioritize documentation and high-level files
    if is_overview:
        print("📚 Detected overview question, prioritizing documentation and high-level files")

        # 1. First, look for README files
        readme_files = []
        for snippet in code_snippets:
            if "filepath" not in snippet or "code" not in snippet:
                continue

            filepath = snippet["filepath"].lower()
            if "readme" in filepath or ".md" in filepath:
                readme_files.append(snippet)

        # 2. Look for entry points (__main__, main.py, app.py)
        entry_points = []
        for snippet in code_snippets:
            if "filepath" not in snippet or "code" not in snippet:
                continue

            filepath = snippet["filepath"].lower()
            code = snippet["code"].lower()

            # Check for main module indicators
            if "__main__" in code or "if __name__ == '__main__'" in code:
                entry_points.append(snippet)
            elif any(name in filepath for name in ["main.py", "app.py", "server.py", "api.py"]):
                entry_points.append(snippet)

        # 3. Find files with the most imports (indicating they're important integration points)
        import_counts = {}
        for snippet in code_snippets:
            if "filepath" not in snippet or "code" not in snippet:
                continue

            filepath = snippet["filepath"]
            code = snippet["code"]

            # Count import statements
            import_count = code.count("import ") + code.count("from ")
            import_counts[filepath] = import_count

        # Get top 5 files with most imports
        import_heavy_files = []
        if import_counts:
            top_import_files = sorted(import_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            for filepath, _ in top_import_files:
                for snippet in code_snippets:
                    if snippet.get("filepath") == filepath and snippet not in import_heavy_files:
                        import_heavy_files.append(snippet)
                        break

        # 4. Build a simplified call graph to find central files
        # This is a simplified approach - in a real implementation, you'd want to use
        # a proper static analysis tool to build a complete call graph
        function_to_file = {}
        file_to_functions = {}

        # First pass: map functions to files
        for snippet in code_snippets:
            if "type" not in snippet or "name" not in snippet or "filepath" not in snippet:
                continue

            if snippet["type"] in ["function", "method"]:
                function_name = snippet["name"]
                filepath = snippet["filepath"]

                function_to_file[function_name] = filepath

                if filepath not in file_to_functions:
                    file_to_functions[filepath] = []
                file_to_functions[filepath].append(function_name)

        # Second pass: count function calls to build a simple call graph
        file_call_counts = {}
        for snippet in code_snippets:
            if "code" not in snippet or "filepath" not in snippet:
                continue

            filepath = snippet["filepath"]
            code = snippet["code"]

            # Initialize call count for this file
            if filepath not in file_call_counts:
                file_call_counts[filepath] = 0

            # Check for function calls
            for function_name in function_to_file:
                # This is a very simplified check - in reality, you'd need more sophisticated parsing
                if function_name + "(" in code:
                    file_call_counts[filepath] += 1

        # Get top 5 files with most function calls
        central_files = []
        if file_call_counts:
            top_call_files = sorted(file_call_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            for filepath, _ in top_call_files:
                for snippet in code_snippets:
                    if snippet.get("filepath") == filepath and snippet not in central_files:
                        central_files.append(snippet)
                        break

        # Combine all the important files, prioritizing in this order:
        # 1. README files
        # 2. Entry points
        # 3. Import-heavy files
        # 4. Central files in the call graph
        overview_files = []

        # Add README files first (up to 2)
        overview_files.extend(readme_files[:2])

        # Add entry points (up to 2)
        for snippet in entry_points[:2]:
            if snippet not in overview_files:
                overview_files.append(snippet)

        # Add import-heavy files (up to 2)
        for snippet in import_heavy_files[:2]:
            if snippet not in overview_files:
                overview_files.append(snippet)

        # Add central files from call graph (up to 2)
        for snippet in central_files[:2]:
            if snippet not in overview_files:
                overview_files.append(snippet)

        # If we still don't have enough files, add from the retrieved indices
        if len(overview_files) < 3:
            for idx in retrieved_indices:
                if len(overview_files) >= 5:  # Limit to 5 files total
                    break

                snippet = code_snippets[idx]
                if snippet not in overview_files:
                    overview_files.append(snippet)

        # Format the overview files
        if overview_files:
            context_parts = []
            context_parts.append("# Project Overview\n\nBased on key files in the codebase:\n")

            for snippet in overview_files:
                if "code" not in snippet:
                    continue

                header = f"From {snippet['filepath']}:"
                code = snippet["code"]

                # Limit code length for overview
                if len(code.split('\n')) > 15:
                    code_lines = code.split('\n')
                    code = '\n'.join(code_lines[:15]) + "\n# ... [code truncated for brevity]"

                # Get language from file extension
                ext = snippet['filepath'].split('.')[-1]
                lang = ext if ext in ['py', 'js', 'java', 'c', 'cpp', 'go', 'rs'] else 'python'

                docstring = snippet.get("docstring", "")
                if not docstring:
                    docstring = "No documentation available"

                # Limit docstring length for overview
                if len(docstring) > 200:
                    docstring = docstring[:200] + "... [docstring truncated for brevity]"

                context_parts.append(f"{header}\n```{lang}\n{code}\n```\n\n{docstring}\n")

            return None, "\n".join(context_parts)
        else:
            # If we couldn't find any good overview files
            return None, "# Project Overview\n\nI couldn't find high-level documentation in the codebase. Try asking about specific components or features instead."

    # For non-overview questions or if no documentation was found, continue with regular retrieval
    # Categorize snippets
    function_snippets = []
    class_snippets = []
    method_snippets = []

    for idx in retrieved_indices:
        snippet = code_snippets[idx]
        # Ensure snippet has a code key
        if "code" not in snippet:
            print(f"⚠️ Warning: Snippet at index {idx} missing 'code' key")
            continue

        # Add importance score
        filepath = snippet.get("filepath", "")
        snippet["importance_score"] = get_file_importance_score(filepath, is_overview)

        if snippet["type"] == "function":
            function_snippets.append(snippet)
        elif snippet["type"] == "class":
            class_snippets.append(snippet)
        elif snippet["type"] == "method":
            method_snippets.append(snippet)

    # Balance the types based on query
    retrieved_items = []
    if "function" in query.lower():
        retrieved_items.extend(function_snippets[:min(15, len(function_snippets))])
        retrieved_items.extend(class_snippets[:min(5, len(class_snippets))])
        retrieved_items.extend(method_snippets[:min(5, len(method_snippets))])
    elif "class" in query.lower():
        retrieved_items.extend(class_snippets[:min(15, len(class_snippets))])
        retrieved_items.extend(function_snippets[:min(5, len(function_snippets))])
        retrieved_items.extend(method_snippets[:min(5, len(method_snippets))])
    else:
        retrieved_items.extend(function_snippets[:min(7, len(function_snippets))])
        retrieved_items.extend(class_snippets[:min(7, len(class_snippets))])
        retrieved_items.extend(method_snippets[:min(6, len(method_snippets))])

    # Verify all items have a code key
    retrieved_items = [item for item in retrieved_items if "code" in item]

    if not retrieved_items:
        print("⚠️ Warning: No valid items retrieved")
        return None, "No relevant code found."

    # Handle large code units and stay within token budget
    processed_items = []
    current_tokens = 0

    for item in retrieved_items:
        # Ensure item has a code key
        if "code" not in item:
            print(f"⚠️ Warning: Item missing 'code' key: {item.keys()}")
            continue

        # Estimate tokens in this snippet
        estimated_tokens = len(item["code"].split()) * 1.3  # Rough estimate

        if estimated_tokens > MAX_TOKENS_PER_SNIPPET:  # This is a large unit
            # For overview questions, use metadata or summarization
            if "what is" in query.lower() or "overview" in query.lower():
                processed_item = extract_code_metadata(item)
                # Ensure metadata has a code key
                if "code" not in processed_item:
                    if "signature" in processed_item:
                        processed_item["code"] = processed_item["signature"]
                    else:
                        processed_item["code"] = f"# {processed_item['type']}: {processed_item['name']}"
                estimated_tokens = 200  # Approximate tokens for metadata
            # For implementation questions, use chunking
            elif "how" in query.lower() or "implementation" in query.lower():
                chunks = chunk_large_code_unit(item)
                processed_item = select_best_chunk(query, chunks)
                estimated_tokens = len(processed_item["code"].split()) * 1.3
            # For other questions, use summarization
            else:
                processed_item = summarize_large_code_unit(item)
                estimated_tokens = len(processed_item["code"].split()) * 1.3

            processed_items.append(processed_item)
        else:
            # Normal sized unit
            processed_items.append(item)

        current_tokens += estimated_tokens

        # Check if we've exceeded our token budget
        if current_tokens >= TOKEN_BUDGET:
            break

    # Apply reranking to the processed candidates
    reranked_items = rerank_snippets(query, processed_items)

    if not reranked_items:
        print("🌍 No relevant local match found. Falling back to generic AI.")
        return None, "No relevant code found."

    # Format the retrieved snippets into a context string
    context_parts = []
    for snippet in reranked_items:
        # Ensure snippet has required keys
        if "code" not in snippet:
            print(f"⚠️ Warning: Reranked snippet missing 'code' key: {snippet.keys()}")
            continue

        # Handle different snippet types (regular, summarized, metadata)
        if "is_summarized" in snippet and snippet["is_summarized"]:
            header = f"{snippet['type'].capitalize()}: {snippet['name']} from {snippet['filepath']} (SUMMARIZED)"
        elif "is_chunk" in snippet and snippet["is_chunk"]:
            header = f"{snippet['type'].capitalize()}: {snippet['name']} from {snippet['filepath']} ({snippet['chunk_info']})"
        elif "signature" in snippet:  # This is a metadata-only snippet
            header = f"{snippet['type'].capitalize()}: {snippet['name']} from {snippet['filepath']} (SIGNATURE ONLY)"
        else:
            header = f"{snippet['type'].capitalize()}: {snippet['name']} from {snippet['filepath']} (lines {snippet['start_line']}-{snippet['end_line']})"

        # Get code, comments and docstring
        code = snippet["code"]
        docstring = snippet.get("docstring", "") if snippet.get("docstring") else "No docstring available"

        # Format the snippet
        context_parts.append(f"{header}\n```python\n{code}\n```\n\nDocstring:\n{docstring}\n")

    return None, "\n".join(context_parts)
# PART 5: INITIALIZATION AND LOADING
def initialize_system():
    """Initialize or load the code understanding system."""
    global faiss_cst_index, faiss_text_index, code_snippets

    if os.path.exists(FAISS_CST_INDEX_FILE) and os.path.exists(FAISS_TEXT_INDEX_FILE) and os.path.exists(CODE_SNIPPETS_FILE):
        print("✅ Loading precomputed FAISS CST and text indexes...")
        faiss_cst_index = faiss.read_index(FAISS_CST_INDEX_FILE)
        faiss_text_index = faiss.read_index(FAISS_TEXT_INDEX_FILE)
        with open(CODE_SNIPPETS_FILE, "rb") as f:
            code_snippets = pickle.load(f)
        print(f"✅ Loaded {len(code_snippets)} code snippets")
    else:
        print("🔄 No FAISS indexes found. Initializing from scratch...")
        index_codebase()

def index_codebase():
    """Index all Python files in the codebase."""
    global code_snippets, faiss_cst_index, faiss_text_index

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
                            faiss_text_index.add(text_embedding.reshape(1, -1))

                            # Create CST embedding
                            cst_text = extract_cst_features(code)
                            cst_embedding = embedding_model.encode(cst_text, convert_to_numpy=True)
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

# PART 6: FLASK APPLICATION
def create_flask_app(testing_mode=False):
    """Create and configure the Flask application."""
    from flask import Flask, request, jsonify, render_template_string
    from openai import OpenAI

    # Initialize the OpenAI client (only if not in testing mode)
    client = None if testing_mode else OpenAI()

    app = Flask(__name__)

    # Add a configuration endpoint to toggle testing mode
    app.config['TESTING_MODE'] = testing_mode

    @app.route('/toggle_testing', methods=['POST'])
    def toggle_testing():
        """Toggle testing mode on/off."""
        app.config['TESTING_MODE'] = not app.config['TESTING_MODE']
        return jsonify({"testing_mode": app.config['TESTING_MODE']})

    # Add a simple GET endpoint for testing
    @app.route('/', methods=['GET'])
    def home():
        return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Code Understanding API</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                h1 { color: #333; }
                .form-group { margin-bottom: 15px; }
                textarea { width: 100%; height: 100px; padding: 8px; }
                button { background: #4CAF50; color: white; padding: 10px 15px; border: none; cursor: pointer; }
                #result { margin-top: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 4px; white-space: pre-wrap; }
                .toggle { margin-top: 20px; }
                .status { font-weight: bold; }
            </style>
        </head>
        <body>
            <h1>Code Understanding API</h1>
            <div class="form-group">
                <label for="query">Enter your query:</label>
                <textarea id="query" placeholder="What does this code do?"></textarea>
            </div>
            <button onclick="sendQuery()">Submit Query</button>
            
            <div class="toggle">
                <span>Testing Mode: <span id="testing-status" class="status">Unknown</span></span>
                <button onclick="toggleTesting()">Toggle Testing Mode</button>
            </div>
            
            <div id="result"></div>

            <script>
            // Check testing mode on page load
            fetch('/testing_status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('testing-status').innerText = data.testing_mode ? 'ON' : 'OFF';
                });
                
            function toggleTesting() {
                fetch('/toggle_testing', {
                    method: 'POST',
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('testing-status').innerText = data.testing_mode ? 'ON' : 'OFF';
                });
            }
            
            function sendQuery() {
                const query = document.getElementById('query').value;
                if (!query) {
                    alert('Please enter a query');
                    return;
                }
                
                document.getElementById('result').innerText = 'Processing...';
                
                fetch('/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ query: query }),
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('result').innerText = data.answer || data.error || 'No response';
                })
                .catch(error => {
                    document.getElementById('result').innerText = 'Error: ' + error;
                });
            }
            </script>
        </body>
        </html>
        """)

    @app.route('/testing_status', methods=['GET'])
    def testing_status():
        """Return the current testing mode status."""
        return jsonify({"testing_mode": app.config['TESTING_MODE']})

    @app.route('/query', methods=['POST'])
    def handle_query():
        data = request.json
        query = data.get('query', '')

        if not query:
            return jsonify({"error": "No query provided"}), 400

        # Retrieve relevant code snippets
        _, context = retrieve_code_snippets(query)

        if not context or context == "No relevant code found.":
            # No relevant code found, use generic response
            response = {"answer": "I couldn't find specific code related to your query."}
        else:
            # Check if we're in testing mode
            if app.config['TESTING_MODE']:
                # Use mock response
                mock_answer = get_mock_response(query, context)
                response = {"answer": mock_answer}
            else:
                # Send to OpenAI with context
                try:
                    prompt = f"""You are an AI assistant helping with code understanding.
                    
                    USER QUERY: {query}
                    
                    RELEVANT CODE SNIPPETS:
                    {context}
                    
                    Based ONLY on the code snippets above, answer the user's query. 
                    If the information isn't in the provided code, say so rather than making things up.
                    Format your response with markdown for code blocks and headings."""

                    # Updated OpenAI API call
                    completion = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that explains code."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=1500
                    )

                    response = {"answer": completion.choices[0].message.content}
                except Exception as e:
                    response = {"error": f"Error generating response: {str(e)}"}

        return jsonify(response)

    return app

# Main execution
if __name__ == "__main__":
    initialize_system()
    # Start in testing mode by default for development
    app = create_flask_app(testing_mode=True)
    app.run(debug=True, port=5000, host='0.0.0.0')
    #Before final improvement