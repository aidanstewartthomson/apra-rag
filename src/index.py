import chromadb
import json
from config import (
    CHROMA_DIR,
    CHUNKS_PATH,
    COLLECTION_NAME,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
)
from openai import OpenAI
from pathlib import Path


def load_chunks(path: Path) -> list[dict]:
    chunks = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    return chunks


def embed_chunks(
    chunks: list[dict], client: OpenAI, batch_size: int = EMBEDDING_BATCH_SIZE
) -> list[list[float]]:
    texts = [chunk["text"] for chunk in chunks]
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)

        embeddings.extend([item.embedding for item in response.data])

    return embeddings


def index_chunks(
    chunks: list[dict], embeddings: list[list[float]], collection: chromadb.Collection
) -> None:
    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [
        {"title": chunk["title"], "section": chunk["section"]} for chunk in chunks
    ]

    collection.add(
        ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents
    )


def main() -> None:
    chunks = load_chunks(CHUNKS_PATH)

    openai_client = OpenAI()
    embeddings = embed_chunks(chunks, openai_client)

    chroma_client = chromadb.PersistentClient(CHROMA_DIR)
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    index_chunks(chunks, embeddings, collection)
    print(f"Indexed {len(chunks)} chunks into {CHROMA_DIR}")


if __name__ == "__main__":
    main()
