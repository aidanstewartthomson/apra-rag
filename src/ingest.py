import json
import polars as pl
import pysbd
import tiktoken
from config import CHUNK_MAX_TOKENS, CHUNKS_PATH, DOCUMENT_DIR, EMBEDDING_MODEL
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
            .with_columns(
                pl.col("text")
                .str.replace_all("•", "")
                .str.replace_all(r"\s+", " ")
                .str.strip_chars()
            )
            .group_by(["title", "section"])
            .agg(pl.col("text").sort_by("idx").str.join(" "))
        )
        frames.append(frame)

    return pl.concat(frames)


def chunk_sentences(
    sentences: list[str], encoding: tiktoken.Encoding, max_tokens: int
) -> list[str]:
    chunks = []
    current = []
    current_tokens = 0

    for sentence in sentences:
        tokens = len(encoding.encode(sentence))

        if current_tokens + tokens <= max_tokens:
            current.append(sentence)
            current_tokens += tokens
        else:
            if current:
                chunks.append(" ".join(current))

            current = [sentence]
            current_tokens = tokens

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_documents(
    documents: pl.DataFrame, max_tokens: int = CHUNK_MAX_TOKENS
) -> list[dict]:
    segmenter = pysbd.Segmenter(clean=True)
    encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)

    chunks = []

    for row in documents.to_dicts():
        sentences = segmenter.segment(row["text"])
        text_chunks = chunk_sentences(sentences, encoding, max_tokens)

        prefix, number = row["title"].split()[:2]
        slug = f"{prefix.lower()}{number.lower()}"

        for i, text in enumerate(text_chunks):
            chunk = dict(row)
            chunk["text"] = text
            chunk["id"] = f"{slug}_{i}"

            chunks.append(chunk)

    return chunks


def save_chunks(chunks: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def main() -> None:
    documents = load_documents(DOCUMENT_DIR)
    print(f"Loaded {len(documents)} documents")

    documents = normalise_documents(documents)
    print(f"Normalised documents into {documents.height} sections")

    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")

    save_chunks(chunks, CHUNKS_PATH)
    print(f"Saved chunks to {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
