import json
from config import CHUNKS_PATH, EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL
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


def main() -> None:
    chunks = load_chunks(CHUNKS_PATH)

    client = OpenAI()
    embeddings = embed_chunks(chunks, client)

    print(embeddings[0])


if __name__ == "__main__":
    main()
