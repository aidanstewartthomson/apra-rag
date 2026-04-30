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
BM25_DIR = DATA_DIR / "bm25"
EVAL_DIR = DATA_DIR / "eval"

MANIFEST_PATH = DATA_DIR / "manifest.csv"
DOCUMENTS_PATH = RAW_DIR / "documents.jsonl"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
QUERIES_PATH = EVAL_DIR / "queries.jsonl"

# models
GENERATION_MODEL = "gpt-5.4-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# ingestion
CHUNK_MAX_TOKENS = 512
CHUNK_MIN_TOKENS = 32

# indexing
DENSE_INDEX_NAME = "dense_index"
SPARSE_INDEX_NAME = "sparse_index"
EMBEDDING_BATCH_SIZE = 256

# retrieval
TOP_K = 10
RRF_K = 60

# evaluation
EVAL_SAMPLE_SIZE = 50
EVAL_SEED = 42

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

EVAL_INSTRUCTIONS = """You are generating evaluation queries for a retrieval system.

Your job is to create realistic user queries that can be answered using only the given chunk.

Rules:
- Generate exactly 3 queries.
- Each query must be answerable using only the provided chunk.
- Do not introduce information not in the chunk.
- Do not copy phrases directly from the chunk.
- Avoid section numbers, paragraph references, or exact headings unless necessary.
- Keep queries concise, natural, and realistic.
- Ensure diversity: one keyword-style, one paraphrased, and one mixed-style query.
- Keyword-style queries should still read like a plausible search, not a list of terms.
- Each query should focus on a single main question.

Structure your response as:
<query 1>
<query 2>
<query 3>
"""
