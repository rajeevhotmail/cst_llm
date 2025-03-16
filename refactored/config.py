import os
import logging
from datetime import datetime

def get_api_key():
    """Get OpenAI API key from environment variable."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return api_key

def setup_logging():
    """Configure logging with timestamp-based file output."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"chat_log_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)