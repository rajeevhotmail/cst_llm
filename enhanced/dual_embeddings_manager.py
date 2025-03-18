import logging
import os
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
import pickle
import ast

def extract_docstring(node):
    """Extract docstring from an AST node"""
    if ast.get_docstring(node):
        return ast.get_docstring(node)
    return ""


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embedding_creation.log'),
        logging.StreamHandler()
    ]
)

class DualEmbeddingsManager:
    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 200

    def __init__(self, base_path="embeddings"):
        logging.info(f"Initializing DualEmbeddingsManager with base path: {base_path}")
        self.base_path = base_path
        self.code_db_path = os.path.join(base_path, "code_embeddings.faiss")
        self.doc_db_path = os.path.join(base_path, "doc_embeddings.faiss")
        self.mappings_path = os.path.join(base_path, "file_mappings.pkl")
        self.chunks_info_path = os.path.join(base_path, "chunks_info.pkl")

        self.code_dimension = 768
        self.doc_dimension = 384

        self.code_model = None
        self.doc_model = None
        self.code_tokenizer = None
        self.chunks_metadata = []
        self.file_mappings = {'code': [], 'doc': []}

        os.makedirs(base_path, exist_ok=True)
        logging.info("Created base directory for embeddings")

        self.initialize_indices()

    def load_models(self):
        logging.info("Loading ML models")
        if not self.code_model:
            self.code_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
            self.code_model = AutoModel.from_pretrained("microsoft/codebert-base")
        if not self.doc_model:
            self.doc_model = SentenceTransformer('all-MiniLM-L6-v2')

    def initialize_indices(self):
        # Add debug logging
        logging.info(f"Checking for existing indices at: {self.code_db_path} and {self.doc_db_path}")

        if os.path.exists(self.code_db_path) and os.path.exists(self.doc_db_path) and os.path.exists(self.mappings_path):
            try:
                self.code_index = faiss.read_index(self.code_db_path)
                self.doc_index = faiss.read_index(self.doc_db_path)
                with open(self.mappings_path, 'rb') as f:
                    self.file_mappings = pickle.load(f)
                # Create set of already processed files
                self.processed_files = {path for paths in self.file_mappings.values() for path, _ in paths}
                logging.info(f"Loaded {len(self.processed_files)} previously processed files")
                return True
            except Exception as e:
                logging.error(f"Error loading existing indices: {e}")

        logging.info("Creating new embedding databases")
        self.code_index = faiss.IndexFlatL2(self.code_dimension)
        self.doc_index = faiss.IndexFlatL2(self.doc_dimension)
        self.file_mappings = {'code': [], 'doc': []}
        self.load_models()
        return False

    def chunk_content(self, content, content_type):
        chunks = []
        chunk_metadata = []

        if content_type == 'code':
            import ast
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        # Get both code and docstring
                        code_chunk = ast.get_source_segment(content, node)
                        docstring = extract_docstring(node)

                        # Combine code and docstring with special separator
                        combined_chunk = f"{code_chunk}\n---DOCSTRING---\n{docstring}"

                        if len(combined_chunk.encode('utf-8')) > self.CHUNK_SIZE:
                            # Handle chunking for large combined content
                            sub_chunks = [combined_chunk[i:i+self.CHUNK_SIZE]
                                        for i in range(0, len(combined_chunk), self.CHUNK_SIZE-self.CHUNK_OVERLAP)]
                            chunks.extend(sub_chunks)
                            for idx, sub_chunk in enumerate(sub_chunks):
                                chunk_metadata.append({
                                    'type': 'code',
                                    'name': node.name,
                                    'has_docstring': bool(docstring),
                                    'sub_chunk': idx,
                                    'size': len(sub_chunk.encode('utf-8'))
                                })
                        else:
                            chunks.append(combined_chunk)
                            chunk_metadata.append({
                                'type': 'code',
                                'name': node.name,
                                'has_docstring': bool(docstring),
                                'size': len(combined_chunk.encode('utf-8'))
                            })
                logging.info(f"Created {len(chunks)} code chunks")
            except SyntaxError as e:
                logging.error(f"Syntax error in code chunking: {e}")

        else:
            # Documentation handling remains the same
            paragraphs = content.split('\n\n')
            current_chunk = ""

            for para in paragraphs:
                if len((current_chunk + para).encode('utf-8')) > self.CHUNK_SIZE:
                    chunks.append(current_chunk)
                    chunk_metadata.append({
                        'type': 'doc',
                        'size': len(current_chunk.encode('utf-8'))
                    })
                    current_chunk = para
                else:
                    current_chunk += "\n\n" + para if current_chunk else para

            if current_chunk:
                chunks.append(current_chunk)
                chunk_metadata.append({
                    'type': 'doc',
                    'size': len(current_chunk.encode('utf-8'))
                })
            logging.info(f"Created {len(chunks)} documentation chunks")

        return chunks, chunk_metadata


    def create_embeddings(self, chunks, content_type):
        logging.info(f"Creating embeddings for {len(chunks)} {content_type} chunks")

        if content_type == 'code':
            if not self.code_model:
                self.load_models()

            embeddings = []
            for chunk in chunks:
                inputs = self.code_tokenizer(chunk, padding=True, truncation=True,
                                           max_length=512, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.code_model(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :].numpy()
                    embeddings.append(embedding[0])
            return np.array(embeddings)

        else:
            if not self.doc_model:
                self.load_models()
            return self.doc_model.encode(chunks)

    def save_indices(self):
        try:
            faiss.write_index(self.code_index, self.code_db_path)
            faiss.write_index(self.doc_index, self.doc_db_path)
            with open(self.mappings_path, 'wb') as f:
                pickle.dump(self.file_mappings, f)
            with open(self.chunks_info_path, 'wb') as f:
                pickle.dump(self.chunks_metadata, f)
            logging.info(f"Successfully saved all indices and metadata to {self.base_path}")
        except Exception as e:
            logging.error(f"Failed to save indices: {e}")
            raise
