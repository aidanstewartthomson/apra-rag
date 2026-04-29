import json
import logging
from pathlib import Path

from chromadb import Collection, PersistentClient
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from rich.logging import RichHandler

from config import (
    CHROMA_DIR,
    DENSE_INDEX_NAME,
    EMBEDDING_MODEL,
)

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger(__name__)


def load_jsonl(jsonl_path: Path) -> list[dict]:
    logger.info("Loading records from %s", jsonl_path)
    records = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    logger.info("Loaded %d records from %s", len(records), jsonl_path)
    return records


def save_jsonl(records: list[dict], jsonl_path: Path) -> None:
    logger.info("Saving %d records to %s", len(records), jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Saved %d records to %s", len(records), jsonl_path)


_client = None
_collection = None


def get_chroma_client() -> PersistentClient:
    global _client

    if _client is None:
        _client = PersistentClient(CHROMA_DIR)

    return _client


def get_chroma_collection() -> Collection:
    global _collection

    if _collection is None:
        client = get_chroma_client()
        embedding_function = OpenAIEmbeddingFunction(model_name=EMBEDDING_MODEL)

        _collection = client.get_or_create_collection(
            DENSE_INDEX_NAME,
            embedding_function=embedding_function,
        )

    return _collection


def rebuild_chroma_collection() -> Collection:
    global _collection

    client = get_chroma_client()

    try:
        client.delete_collection(DENSE_INDEX_NAME)
    except Exception:
        pass

    _collection = None
    return get_chroma_collection()
