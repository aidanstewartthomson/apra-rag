import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DOCUMENT_DIR = Path("data/raw")
CHUNKS_PATH = Path("data/processed/chunks.jsonl")
CHROMA_DIR = Path("data/chroma")

CHUNK_MAX_TOKENS = 400

COLLECTION_NAME = "chunks"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BATCH_SIZE = 200

TOP_K = 5

GENERATION_MODEL = "gpt-5.4-mini"
