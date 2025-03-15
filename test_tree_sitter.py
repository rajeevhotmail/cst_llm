from dotenv import load_dotenv
import os

load_dotenv("openai_key.env")

api_key = os.getenv("OPENAI_API_KEY")
print(f"Loaded API Key: {api_key[:5]}...")  # Print first 5 characters only
