import random

from openai import OpenAI

from config import (
    CHUNKS_PATH,
    EVAL_INSTRUCTIONS,
    EVAL_SAMPLE_SIZE,
    EVAL_SEED,
    GENERATION_MODEL,
    QUERIES_PATH,
)
from utils import load_jsonl, save_jsonl


def sample_chunks(
    chunks: list[dict], num_samples: int = EVAL_SAMPLE_SIZE, seed: int | None = None
) -> list[dict]:
    rng = random.Random(seed) if seed is not None else random
    samples = rng.sample(chunks, num_samples)

    return samples


def generate_dataset(chunks: list[dict], client: OpenAI) -> list[dict]:
    records = []
    for chunk in chunks:
        prompt = f"Chunk:\n{chunk['text']}"
        response = client.responses.create(
            model=GENERATION_MODEL, instructions=EVAL_INSTRUCTIONS, input=prompt
        )

        queries = [q.strip() for q in response.output_text.split("\n") if q.strip()]
        for i, query in enumerate(queries, start=1):
            records.append(
                {"id": f"{chunk['id']}_q{i}", "query": query, "chunk_id": chunk["id"]}
            )

    return records


def main() -> None:
    chunks = load_jsonl(CHUNKS_PATH)
    samples = sample_chunks(chunks, seed=EVAL_SEED)

    client = OpenAI()
    dataset = generate_dataset(samples, client)

    save_jsonl(dataset, QUERIES_PATH)


if __name__ == "__main__":
    main()
