import random

from config import CHUNKS_PATH, EVAL_SAMPLE_SIZE, EVAL_SEED
from utils import load_jsonl


def sample_chunks(
    chunks: list[dict], num_samples: int = EVAL_SAMPLE_SIZE, seed: int | None = None
) -> list[dict]:
    rng = random.Random(seed) if seed else random
    samples = rng.sample(chunks, num_samples)

    return samples


def main() -> None:
    chunks = load_jsonl(CHUNKS_PATH)
    samples = sample_chunks(chunks, seed=EVAL_SEED)

    for sample in samples[:5]:
        print(f"\n{sample["text"]}")


if __name__ == "__main__":
    main()
