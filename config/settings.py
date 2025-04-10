import os
from dotenv import load_dotenv
load_dotenv()

# API keys and model configuration
ARXIV_API_KEY = os.getenv("ARXIV_API_KEY", "")       # Arxiv may not require this; include if needed.
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
VECTORSTORE_SAVE_PATH = os.getenv("VECTORSTORE_SAVE_PATH", "./embeddings")
STATIC_VS_SAVE_PATH = "./emails_bitcoindev.jsonl"
EMAILS_FILE = os.path.join(os.path.dirname(__file__), "..", "emails_bitcoindev.jsonl")