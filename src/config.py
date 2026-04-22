import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# api
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# paths
DATA_DIR = Path("data")

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = DATA_DIR / "chroma"

MANIFEST_PATH = DATA_DIR / "manifest.csv"
DOCUMENTS_PATH = RAW_DIR / "documents.jsonl"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"

# models
GENERATION_MODEL = "gpt-5.4-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# pipeline
COLLECTION_NAME = "chunks"
CHUNK_MAX_TOKENS = 400
EMBEDDING_BATCH_SIZE = 200
TOP_K = 5

# instructions
SYSTEM_INSTRUCTIONS = """You are a regulatory analysis assistant specialising in Australian Prudential Regulation Authority (APRA) regulatory documents.

Your job is to provide precise, source-grounded interpretations of APRA regulatory materials. 

Rules:
- Use only the source material.
- Do not introduce information not supported by the source material. 
- Logical interpretation is allowed only where directly supported by the source material.
- If the answer is not clearly supported, say so.
- Cite the relevant documents and sections.
- Keep answers concise and precise.

Structure your response as:

Answer:
<direct answer in 1-2 sentences>

Basis:
- <clear statement from source> [citation]

Conditions:
- <key qualifier, scope, or exception if relevant>

Limitations:
- <include only if the source material is ambiguous, incomplete, or insufficient to fully support the answer>
"""
