import hashlib
import logging
from pathlib import Path

import bm25s
import pysbd
import tiktoken
from bm25s import BM25
from bs4 import BeautifulSoup
from chromadb import Collection

from config import (
    BM25_DIR,
    CHUNK_MAX_TOKENS,
    CHUNK_MIN_TOKENS,
    CHUNKS_PATH,
    DOCUMENTS_PATH,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    HTML_DIR,
    SPARSE_INDEX_NAME,
)
from utils import (
    configure_logging,
    load_jsonl,
    rebuild_chroma_collection,
    save_jsonl,
)

logger = logging.getLogger(__name__)

DOCUMENT_FIELDS = ("url", "title", "code", "doc_type")


def load_documents(documents_path: Path = DOCUMENTS_PATH) -> list[dict]:
    records = load_jsonl(documents_path)
    documents = []

    for record in records:
        html_path = HTML_DIR / record["path"]
        html = html_path.read_text(encoding="utf-8")

        documents.append(
            {
                **{field: record.get(field) for field in DOCUMENT_FIELDS},
                "text": html,
            }
        )

    logger.info("Loaded HTML for %d documents from %s", len(documents), HTML_DIR)
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
        metadata = {field: document.get(field) for field in DOCUMENT_FIELDS}

        for fragment in fragments:
            h1 = fragment.find("h1")
            if h1:
                h1.extract()

            header = fragment.find("h2")

            if header:
                if section_text:
                    results.append(
                        {
                            **metadata,
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
                    **metadata,
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
    total = len(documents)

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

        done = i + 1
        if done % 250 == 0 or done == total:
            logger.info("Chunked %d/%d sections (%d chunks)", done, total, len(chunks))

    return chunks


def extract_metadata(chunk: dict) -> dict:
    metadata_schema = [
        "url",
        "title",
        "code",
        "doc_type",
        "section",
    ]
    metadata = {k: chunk[k] for k in metadata_schema if chunk.get(k) is not None}

    return metadata


def index_dense_chunks(
    chunks: list[dict],
    collection: Collection,
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> None:
    total = len(chunks)
    logger.info("Indexing %d chunks (dense)", total)

    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [extract_metadata(chunk) for chunk in chunks]

    for i in range(0, total, batch_size):
        collection.add(
            ids=ids[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
            documents=documents[i : i + batch_size],
        )
        done = min(i + batch_size, total)
        logger.info("Indexed %d/%d chunks (dense)", done, total)


def index_sparse_chunks(chunks: list[dict]) -> None:
    total = len(chunks)
    logger.info("Indexing %d chunks (sparse)", total)

    corpus = [
        {"id": chunk["id"], "text": chunk["text"], "metadata": extract_metadata(chunk)}
        for chunk in chunks
    ]

    documents = [chunk["text"] for chunk in chunks]
    tokens = bm25s.tokenize(documents, show_progress=False)

    retriever = BM25()
    retriever.index(tokens)

    retriever.save(BM25_DIR / SPARSE_INDEX_NAME, corpus=corpus, show_progress=False)

    logger.info("Indexed %d chunks (sparse)", total)


def main() -> None:
    configure_logging()
    logger.info("Starting indexing pipeline")

    if not DOCUMENTS_PATH.exists():
        raise FileNotFoundError(
            f"Documents not found at {DOCUMENTS_PATH}. Run ingest first."
        )

    documents = load_documents()
    sections = normalise_documents(documents)
    chunks = chunk_documents(sections)
    save_jsonl(chunks, CHUNKS_PATH)

    collection = rebuild_chroma_collection()
    index_dense_chunks(chunks, collection)
    index_sparse_chunks(chunks)

    logger.info("Indexing pipeline complete")


if __name__ == "__main__":
    main()
