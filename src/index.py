import json
from pathlib import Path


def load_chunks(path: Path) -> list[dict]:
    chunks = []

    with path.open("r") as f:
        for line in f:
            chunks.append(json.loads(line))

    return chunks


def main():
    input_path = Path("data/processed/chunks.jsonl")
    chunks = load_chunks(input_path)

    print(chunks[0])


if __name__ == "__main__":
    main()
