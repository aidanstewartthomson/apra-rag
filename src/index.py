import logging

import bm25s
from bm25s import BM25
from chromadb import Collection
from rich.logging import RichHandler

from config import (
    BM25_DIR,
    CHUNKS_PATH,
    DENSE_INDEX_NAME,
    EMBEDDING_BATCH_SIZE,
    SPARSE_INDEX_NAME,
)
from utils import (
    load_jsonl,
    rebuild_chroma_collection,
)

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])

# silence noisy logs
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("bm25s").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def extract_metadata(chunk: dict) -> dict:
    metadata_schema = [
        "url",
        "title",
        "description",
        "industry",
        "pillar",
        "sub_pillar",
        "code",
        "effective_date",
        "section",
    ]
    metadata = {k: chunk[k] for k in metadata_schema if chunk.get(k) is not None}

    return metadata


def index_dense_chunks(
    chunks: list[dict],
    collection: Collection,
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> None:
    logger.info(
        "Indexing %d chunks into dense index '%s'", len(chunks), DENSE_INDEX_NAME
    )

    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [extract_metadata(chunk) for chunk in chunks]

    for i in range(0, len(chunks), batch_size):
        collection.add(
            ids=ids[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
            documents=documents[i : i + batch_size],
        )

    logger.info(
        "Indexed %d chunks into dense index '%s'", len(chunks), DENSE_INDEX_NAME
    )


def index_sparse_chunks(chunks: list[dict]) -> None:
    logger.info(
        "Indexing %d chunks into sparse index '%s'",
        len(chunks),
        SPARSE_INDEX_NAME,
    )

    corpus = [
        {"id": chunk["id"], "text": chunk["text"], "metadata": extract_metadata(chunk)}
        for chunk in chunks
    ]

    documents = [chunk["text"] for chunk in chunks]
    tokens = bm25s.tokenize(documents)

    retriever = BM25()
    retriever.index(tokens)

    retriever.save(BM25_DIR / SPARSE_INDEX_NAME, corpus=corpus, show_progress=False)

    logger.info(
        "Indexed %d chunks into sparse index '%s'",
        len(chunks),
        SPARSE_INDEX_NAME,
    )


def main() -> None:
    logger.info("Starting indexing pipeline")

    chunks = load_jsonl(CHUNKS_PATH)
    collection = rebuild_chroma_collection()

    index_dense_chunks(chunks, collection)
    index_sparse_chunks(chunks)

    logger.info("Indexing pipeline complete")


if __name__ == "__main__":
    main()
