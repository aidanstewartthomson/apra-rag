import json
import polars as pl
from pathlib import Path


def load_documents(directory: Path) -> list[pl.DataFrame]:
    documents = []

    for path in directory.glob("*.csv"):
        header = pl.read_csv(path, has_header=False, n_rows=1)
        title = header[0, 1]

        df = pl.read_csv(path, skip_lines=3)
        df = df.with_columns(pl.lit(title).alias("title"))
        documents.append(df)

    return documents


def normalise_documents(documents: list[pl.DataFrame]) -> pl.DataFrame:
    frames = []

    for df in documents:
        frame = (
            df.with_row_index("idx")
            .drop("Fragment ID")
            .rename({"Heading": "section", "Content": "text"})
            .with_columns(pl.col(["section", "text"]).replace("", None))
            .with_columns(pl.col("section").forward_fill())
            .filter(pl.col("text").is_not_null())
            .group_by(["title", "section"])
            .agg(pl.col("text").sort_by("idx").str.join(" "))
        )
        frames.append(frame)

    return pl.concat(frames)


def chunk_documents(documents: pl.DataFrame) -> list[dict]:
    chunks = documents.to_dicts()

    for i, chunk in enumerate(chunks):
        chunk["chunk_id"] = i

    return chunks


def save_chunks(chunks: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def main():
    input_dir = Path("data/raw")
    output_path = Path("data/processed/chunks.jsonl")

    documents = load_documents(input_dir)
    print(f"Loaded {len(documents)} documents")

    documents = normalise_documents(documents)
    print(f"Normalised documents into {documents.height} sections")

    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")

    save_chunks(chunks, output_path)
    print(f"Saved chunks to {output_path}")


if __name__ == "__main__":
    main()
