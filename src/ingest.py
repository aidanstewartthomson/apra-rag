import hashlib
import logging
from pathlib import Path

import polars as pl
import pysbd
import requests
import tiktoken
from bs4 import BeautifulSoup
from rich.logging import RichHandler

from config import (
    CHUNK_MAX_TOKENS,
    CHUNK_MIN_TOKENS,
    CHUNKS_PATH,
    DOCUMENTS_PATH,
    EMBEDDING_MODEL,
    MANIFEST_PATH,
)
from utils import load_jsonl, save_jsonl

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger(__name__)


def load_manifest(manifest_path: Path) -> list[dict]:
    logger.info("Loading manifest from %s", manifest_path)

    df = pl.read_csv(manifest_path).filter(
        (pl.col("status") == "Current")
        & pl.col("url").is_not_null()
        # urls with "qa.handbook" aren't accessible without a login
        & ~pl.col("url").str.contains("qa.handbook")
    )

    manifest = df.to_dicts()
    logger.info("Loaded %d manifest entries after filtering", len(manifest))

    return manifest


def fetch_documents(manifest_path: Path) -> list[dict]:
    rows = load_manifest(manifest_path)
    documents = []

    logger.info("Fetching %d documents", len(rows))

    for row in rows:
        url = row.get("url")

        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.RequestException:
            logger.exception("Failed to fetch document from %s", url)
            continue

        documents.append(
            {
                "url": url,
                "title": row.get("title"),
                "description": row.get("description"),
                "doc_type": row.get("doc_type"),
                "industry": row.get("industry"),
                "pillar": row.get("pillar"),
                "sub_pillar": row.get("sub_pillar"),
                "code": row.get("code"),
                "effective_date": row.get("effective_date"),
                "section": None,
                "text": response.text,
            }
        )

    logger.info("Fetched %d/%d documents successfully", len(documents), len(rows))
    return documents


def normalise_documents(documents: list[dict]) -> list[dict]:
    logger.info("Normalising %d documents", len(documents))
    results = []

    for document in documents:
        soup = BeautifulSoup(document["text"], "lxml")

        # fragments are how APRA document web pages are structured
        fragments = soup.find_all(class_="fragment")

        current_section = None
        section_text = []

        for fragment in fragments:
            h1 = fragment.find("h1")
            if h1:
                h1.extract()

            header = fragment.find("h2")

            if header:
                if section_text:
                    results.append(
                        {
                            **document,
                            "section": current_section,
                            "text": " ".join(section_text),
                        }
                    )

                current_section = header.get_text(strip=True)
                header.extract()
                section_text = []

            text = fragment.get_text(" ", strip=True)

            if text:
                section_text.append(text)

        if section_text:
            results.append(
                {
                    **document,
                    "section": current_section,
                    "text": " ".join(section_text),
                }
            )

    logger.info(
        "Normalised %d documents into %d sections", len(documents), len(results)
    )
    return results


def chunk_sentences(
    sentences: list[str], encoding: tiktoken.Encoding, max_tokens: int
) -> list[str]:
    chunks = []
    current = []
    current_tokens = 0

    for sentence in sentences:
        tokens = len(encoding.encode(sentence))

        if current_tokens + tokens <= max_tokens:
            current.append(sentence)
            current_tokens += tokens
        else:
            if current:
                chunks.append(" ".join(current))

            current = [sentence]
            current_tokens = tokens

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_documents(
    documents: list[dict],
    max_tokens: int = CHUNK_MAX_TOKENS,
    min_tokens: int = CHUNK_MIN_TOKENS,
) -> list[dict]:
    logger.info(
        "Chunking %d sections (max_tokens=%d, min_tokens=%d)",
        len(documents),
        max_tokens,
        min_tokens,
    )

    segmenter = pysbd.Segmenter(clean=True)
    encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)

    chunks = []

    for i, document in enumerate(documents):
        sentences = segmenter.segment(document["text"])
        chunk_texts = chunk_sentences(sentences, encoding, max_tokens)

        for j, text in enumerate(chunk_texts):
            token_count = len(encoding.encode(text))
            if token_count < min_tokens:
                continue

            key = f"{document['url']}{i}{j}"
            chunk_id = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

            chunks.append(
                {
                    "id": chunk_id,
                    **document,
                    "text": text,
                }
            )

    logger.info("Chunked %d sections into %d chunks", len(documents), len(chunks))
    return chunks


def main() -> None:
    logger.info("Starting ingestion pipeline")

    if DOCUMENTS_PATH.exists():
        logger.info("Using existing documents from %s", DOCUMENTS_PATH)
        documents = load_jsonl(DOCUMENTS_PATH)
    else:
        logger.info("No existing documents found")
        documents = fetch_documents(MANIFEST_PATH)
        save_jsonl(documents, DOCUMENTS_PATH)

    documents = normalise_documents(documents)
    chunks = chunk_documents(documents)
    save_jsonl(chunks, CHUNKS_PATH)

    logger.info("Ingestion pipeline complete")


if __name__ == "__main__":
    main()
