import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DOCUMENT_DIR = Path("data/raw")
CHUNKS_PATH = Path("data/processed/chunks.jsonl")

EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_MAX_TOKENS = 400
