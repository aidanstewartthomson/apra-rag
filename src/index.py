import logging

import bm25s
from bm25s import BM25
from chromadb import Collection
from config import (
    BM25_DIR,
    CHUNKS_PATH,
    DENSE_INDEX_NAME,
    EMBEDDING_BATCH_SIZE,
    SPARSE_INDEX_NAME,
)
from utils import (
    configure_logging,
    load_jsonl,
    rebuild_chroma_collection,
)

logger = logging.getLogger(__name__)


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

    chunks = load_jsonl(CHUNKS_PATH)
    collection = rebuild_chroma_collection()

    index_dense_chunks(chunks, collection)
    index_sparse_chunks(chunks)

    logger.info("Indexing pipeline complete")


if __name__ == "__main__":
    main()
