import json
from config import CHUNKS_PATH
from pathlib import Path


def load_chunks(path: Path) -> list[dict]:
    chunks = []

    with path.open("r") as f:
        for line in f:
            chunks.append(json.loads(line))

    return chunks


def main():
    chunks = load_chunks(CHUNKS_PATH)

    print(chunks[0])


if __name__ == "__main__":
    main()
