import random

import chromadb
from openai import OpenAI

from config import (
    CHROMA_DIR,
    CHUNKS_PATH,
    COLLECTION_NAME,
    EVAL_INSTRUCTIONS,
    EVAL_SAMPLE_SIZE,
    EVAL_SEED,
    GENERATION_MODEL,
    QUERIES_PATH,
    TOP_K,
)
from retrieve import retrieve_chunks
from utils import load_jsonl, save_jsonl


def sample_chunks(
    chunks: list[dict], n_results: int = EVAL_SAMPLE_SIZE, seed: int | None = None
) -> list[dict]:
    rng = random.Random(seed) if seed is not None else random
    samples = rng.sample(chunks, n_results)

    return samples


def generate_queries(chunks: list[dict], client: OpenAI) -> list[dict]:
    records = []
    for chunk in chunks:
        prompt = f"Chunk:\n{chunk['text']}"
        response = client.responses.create(
            model=GENERATION_MODEL, instructions=EVAL_INSTRUCTIONS, input=prompt
        )

        queries = [q.strip() for q in response.output_text.split("\n") if q.strip()]
        for i, query in enumerate(queries, start=1):
            records.append(
                {"id": f"{chunk['id']}_q{i}", "text": query, "chunk_id": chunk["id"]}
            )

    return records


def evaluate_queries(
    queries: list[dict],
    collection: chromadb.Collection,
    client: OpenAI,
    n_results: int = TOP_K,
) -> dict:
    reciprocal_ranks = []
    hits = 0

    for query in queries:
        chunks = retrieve_chunks(query["text"], collection, client, n_results=n_results)

        reciprocal_rank = 0
        for rank, chunk in enumerate(chunks, start=1):
            if chunk["id"] == query["chunk_id"]:
                reciprocal_rank = 1 / rank
                hits += 1
                break

        reciprocal_ranks.append(reciprocal_rank)

    total = len(queries)

    return {
        "recall": hits / total,
        "mrr": sum(reciprocal_ranks) / total,
        "top_k": n_results,
        "hits": hits,
        "total": total,
    }


def main() -> None:
    chroma_client = chromadb.PersistentClient(CHROMA_DIR)
    collection = chroma_client.get_collection(COLLECTION_NAME)

    openai_client = OpenAI()

    if QUERIES_PATH.exists():
        queries = load_jsonl(QUERIES_PATH)
    else:
        chunks = load_jsonl(CHUNKS_PATH)
        samples = sample_chunks(chunks, seed=EVAL_SEED)
        queries = generate_queries(samples, openai_client)

        save_jsonl(queries, QUERIES_PATH)

    evaluation = evaluate_queries(queries, collection, openai_client)

    for key, value in evaluation.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
