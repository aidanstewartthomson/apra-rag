import logging

from chromadb import Collection
from rich.logging import RichHandler

from config import (
    CHUNKS_PATH,
    COLLECTION_NAME,
    EMBEDDING_BATCH_SIZE,
)
from utils import get_chroma_client, get_chroma_collection, load_jsonl

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])

# remove annoying openai info logs
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

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


def index_chunks(
    chunks: list[dict],
    collection: Collection,
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> None:
    logger.info("Indexing %d chunks into collection '%s'", len(chunks), collection.name)

    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [extract_metadata(chunk) for chunk in chunks]

    for i in range(0, len(chunks), batch_size):
        collection.add(
            ids=ids[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
            documents=documents[i : i + batch_size],
        )

    logger.info("Indexed %d chunks into collection '%s'", len(chunks), collection.name)


def main() -> None:
    logger.info("Starting indexing pipeline")

    chroma_client = get_chroma_client()
    chunks = load_jsonl(CHUNKS_PATH)

    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        logger.info("Deleted existing collection '%s'", COLLECTION_NAME)
    except Exception:
        logger.info("Collection '%s' did not already exist", COLLECTION_NAME)

    collection = get_chroma_collection()
    index_chunks(chunks, collection)

    logger.info("Indexing pipeline complete")


if __name__ == "__main__":
    main()
