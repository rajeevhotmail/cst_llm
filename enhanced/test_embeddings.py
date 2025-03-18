import os
import logging
from dual_embeddings_manager import DualEmbeddingsManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_embeddings.log'),
        logging.StreamHandler()
    ]
)
def is_text_file(file_path):
    # Skip git files and binary files
    if '.git' in file_path:
        return False
    # Only process these extensions
    valid_extensions = {'.py', '.md', '.txt', '.yaml', '.yml', '.json'}
    return any(file_path.endswith(ext) for ext in valid_extensions)

def test_db_creation():
    source_code_path = "../source_code"
    logging.info(f"Starting embedding creation test with source code from: {source_code_path}")

    manager = DualEmbeddingsManager()

    for root, _, files in os.walk(source_code_path):
        for file in files:
            file_path = os.path.join(root, file)
            # Skip if already processed
            if hasattr(manager, 'processed_files') and file_path in manager.processed_files:
                logging.info(f"Skipping already processed file: {file_path}")
                continue
            if not is_text_file(file_path):
                logging.info(f"Skipping non-text file: {file_path}")
                continue

            logging.info(f"Processing file: {file_path}")

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Skip empty files
                if not content.strip():
                    logging.info(f"Skipping empty file: {file_path}")
                    continue

                is_code = file.endswith('.py')
                content_type = 'code' if is_code else 'doc'

                # Create chunks
                chunks, chunk_metadata = manager.chunk_content(content, content_type)

                # Skip if no chunks were created
                if not chunks:
                    logging.info(f"No chunks created for {file_path}, skipping")
                    continue

                # Create embeddings
                embeddings = manager.create_embeddings(chunks, content_type)
                logging.info(f"Created embeddings of shape {embeddings.shape}")

                # Add to appropriate index
                if is_code:
                    manager.code_index.add(embeddings)
                else:
                    manager.doc_index.add(embeddings)

                # Update mappings
                manager.file_mappings[content_type].extend([(file_path, i) for i in range(len(chunks))])

            except UnicodeDecodeError:
                logging.warning(f"Could not read file as text: {file_path}")
                continue

    manager.save_indices()
    return manager


if __name__ == "__main__":
    manager = test_db_creation()
