import logging
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
from utils import configure_logging, load_jsonl, save_jsonl

logger = logging.getLogger(__name__)


def sample_chunks(
    chunks: list[dict], n_results: int = EVAL_SAMPLE_SIZE, seed: int | None = None
) -> list[dict]:
    rng = random.Random(seed) if seed is not None else random
    samples = rng.sample(chunks, n_results)

    logger.info("Sampled %d/%d chunks", n_results, len(chunks))
    return samples


def make_queries(chunks: list[dict], client: OpenAI) -> list[dict]:
    records = []
    total = len(chunks)
    logger.info("Building queries for %d chunks", total)

    for i, chunk in enumerate(chunks, start=1):
        prompt = f"Chunk:\n{chunk['text']}"
        response = client.responses.create(
            model=GENERATION_MODEL, instructions=EVAL_INSTRUCTIONS, input=prompt
        )

        queries = [q.strip() for q in response.output_text.split("\n") if q.strip()]

        # ensure only three queries per chunk
        for j, query in enumerate(queries[:3], start=1):
            records.append(
                {"id": f"{chunk['id']}_q{j}", "text": query, "chunk_id": chunk["id"]}
            )

        if i % 10 == 0 or i == total:
            logger.info("Built %d/%d chunks (%d queries)", i, total, len(records))

    return records


def main() -> None:
    configure_logging()
    logger.warning(
        "Rebuilding queries creates a new benchmark; previous eval scores are not comparable"
    )

    chunks = load_jsonl(CHUNKS_PATH)
    samples = sample_chunks(chunks, seed=EVAL_SEED)
    queries = make_queries(samples, OpenAI())
    save_jsonl(queries, QUERIES_PATH)


if __name__ == "__main__":
    main()
