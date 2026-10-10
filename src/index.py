import hashlib
import logging
from pathlib import Path

import bm25s
from bm25s import BM25
from bs4 import BeautifulSoup
from chromadb import Collection
from langchain_text_splitters import (
    HTMLHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from config import (
    BM25_DIR,
    CHUNK_MAX_SIZE,
    CHUNK_MIN_SIZE,
    CHUNKS_PATH,
    DOCUMENTS_PATH,
    EMBEDDING_BATCH_SIZE,
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

HEADERS_TO_SPLIT_ON = [(f"h{i}", f"Header {i}") for i in range(1, 7)]
HEADER_KEYS = [name for _, name in HEADERS_TO_SPLIT_ON]


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


INLINE_TAGS = ("a", "i", "em", "b", "strong", "span", "u", "sup", "sub")


def extract_content(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    root = soup.find(class_="fragments") or soup

    # drop footnotes
    for tag in root.find_all(class_=["footnote", "footnote-reference"]):
        tag.decompose()

    # LangChain HTML splitters treat inline tags as separate elements and
    # pull their text out of sentence order; unwrap so text stays in place.
    for tag in root.find_all(INLINE_TAGS):
        tag.unwrap()

    return str(root)


def section_from_metadata(metadata: dict) -> str | None:
    for key in reversed(HEADER_KEYS):
        if metadata.get(key):
            return metadata[key]
    return None


def is_header_only_split(text: str, metadata: dict) -> bool:
    return text in {metadata[key] for key in HEADER_KEYS if metadata.get(key)}


def chunk_documents(documents: list[dict]) -> list[dict]:
    logger.info(
        "Chunking %d documents (min=%d, max=%d)",
        len(documents),
        CHUNK_MIN_SIZE,
        CHUNK_MAX_SIZE,
    )

    header_splitter = HTMLHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_MAX_SIZE,
        chunk_overlap=0,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    total = len(documents)

    for i, document in enumerate(documents):
        splits = header_splitter.split_text(extract_content(document["text"]))
        metadata = {field: document.get(field) for field in DOCUMENT_FIELDS}

        for j, split in enumerate(splits):
            text = split.page_content.strip()
            if (
                not text
                or is_header_only_split(text, split.metadata)
                or len(text) < CHUNK_MIN_SIZE
            ):
                continue

            section = section_from_metadata(split.metadata)
            parts = (
                size_splitter.split_text(text) if len(text) > CHUNK_MAX_SIZE else [text]
            )

            for k, part in enumerate(parts):
                part = part.strip()
                if len(part) < CHUNK_MIN_SIZE:
                    continue

                key = f"{document['url']}{i}{j}{k}"
                chunk_id = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

                chunks.append(
                    {
                        "id": chunk_id,
                        **metadata,
                        "section": section,
                        "text": part,
                    }
                )

        done = i + 1
        if done % 50 == 0 or done == total:
            logger.info("Chunked %d/%d documents (%d chunks)", done, total, len(chunks))

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
    chunks = chunk_documents(documents)
    save_jsonl(chunks, CHUNKS_PATH)

    collection = rebuild_chroma_collection()
    index_dense_chunks(chunks, collection)
    index_sparse_chunks(chunks)

    logger.info("Indexing pipeline complete")


if __name__ == "__main__":
    main()
