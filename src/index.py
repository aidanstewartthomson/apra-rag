import chromadb
import json
import logging
from config import (
    CHROMA_DIR,
    CHUNKS_PATH,
    COLLECTION_NAME,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
)
from openai import OpenAI
from pathlib import Path
from rich.logging import RichHandler

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def load_chunks(chunks_path: Path) -> list[dict]:
    logger.info("Loading chunks from %s", chunks_path)
    chunks = []

    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    logger.info("Loaded %d chunks from %s", len(chunks), chunks_path)
    return chunks


def embed_chunks(
    chunks: list[dict], client: OpenAI, batch_size: int = EMBEDDING_BATCH_SIZE
) -> list[list[float]]:
    logger.info("Embedding %d chunks with batch_size=%d", len(chunks), batch_size)

    texts = [chunk["text"] for chunk in chunks]
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)

        embeddings.extend([item.embedding for item in response.data])

    logger.info("Embedded %d chunks into %d vectors", len(chunks), len(embeddings))
    return embeddings


def index_chunks(
    chunks: list[dict], embeddings: list[list[float]], collection: chromadb.Collection
) -> None:
    logger.info("Indexing %d chunks into collection '%s'", len(chunks), collection.name)

    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]

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
    metadatas = [
        {k: chunk[k] for k in metadata_schema if chunk.get(k) is not None}
        for chunk in chunks
    ]

    collection.add(
        ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents
    )

    logger.info("Indexed %d chunks into collection '%s'", len(chunks), collection.name)


def main() -> None:
    logger.info("Starting indexing pipeline")

    chunks = load_chunks(CHUNKS_PATH)

    openai_client = OpenAI()
    embeddings = embed_chunks(chunks, openai_client)

    chroma_client = chromadb.PersistentClient(CHROMA_DIR)
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    index_chunks(chunks, embeddings, collection)

    logger.info("Indexing pipeline complete")


if __name__ == "__main__":
    main()
