import json
import polars as pl
import pysbd
import requests
import tiktoken

from config import (
    CHUNK_MAX_TOKENS,
    CHUNKS_PATH,
    DOCUMENTS_PATH,
    MANIFEST_PATH,
    EMBEDDING_MODEL,
)
from pathlib import Path


def load_manifest(manifest_path: Path) -> list[dict]:
    df = pl.read_csv(manifest_path).filter(
        (pl.col("status") == "Current")
        & pl.col("url").is_not_null()
        # urls with "qa.handbook" aren't accessible without a login
        & ~pl.col("url").str.contains("qa.handbook")
    )

    return df.to_dicts()


def fetch_documents(manifest_path: Path) -> list[dict]:
    rows = load_manifest(MANIFEST_PATH)
    documents = []

    for row in rows:
        url = row.get("url")
        response = requests.get(url)

        documents.append(
            {
                "url": url,
                "title": row.get("title"),
                "description": row.get("description"),
                "doc_type": row.get("doc_type"),
                "code": row.get("code"),
                "industry": row.get("industry"),
                "pillar": row.get("pillar"),
                "sub_pillar": row.get("sub-pillar"),
                "effective_date": row.get("effective_date"),
                "html": response.text,
            }
        )

    return documents


def save_documents(documents: list[dict], documents_path: Path) -> None:
    documents_path.parent.mkdir(parents=True, exist_ok=True)

    with documents_path.open("w", encoding="utf-8") as f:
        for document in documents:
            f.write(json.dumps(document) + "\n")


def load_documents(documents_path: Path) -> list[dict]:
    documents = []

    with documents_path.open("r", encoding="utf-8") as f:
        for line in f:
            documents.append(json.loads(line))

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
    rows = documents.to_dicts()

    for row in rows:
        sentences = segmenter.segment(row["text"])
        chunk_texts = chunk_sentences(sentences, encoding, max_tokens)

        for text in chunk_texts:
            chunk = {
                "id": str(len(chunks)),
                "title": row["title"],
                "section": row["section"],
                "text": text,
            }
            chunks.append(chunk)

    return chunks


def save_chunks(chunks: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def main() -> None:
    if DOCUMENTS_PATH.exists():
        documents = load_documents(DOCUMENTS_PATH)
        print(documents[0])
    else:
        documents = fetch_documents(MANIFEST_PATH)
        print(f"Fetched {len(documents)} documents")

        save_documents(documents, DOCUMENTS_PATH)
        print(f"Saved {len(documents)} to {DOCUMENTS_PATH}")

    # documents = normalise_documents(documents)
    # print(f"Normalised documents into {documents.height} sections")

    # chunks = chunk_documents(documents)
    # print(f"Created {len(chunks)} chunks")

    # save_chunks(chunks, CHUNKS_PATH)
    # print(f"Saved chunks to {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
