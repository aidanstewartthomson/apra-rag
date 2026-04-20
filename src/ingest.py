import json
import polars as pl
import pysbd
import requests
import tiktoken

from bs4 import BeautifulSoup
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
                "industry": row.get("industry"),
                "pillar": row.get("pillar"),
                "sub_pillar": row.get("sub_pillar"),
                "code": row.get("code"),
                "effective_date": row.get("effective_date"),
                "section": None,
                "text": response.text,
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


def normalise_documents(documents: list[dict]) -> list[dict]:
    results = []

    for document in documents:
        soup = BeautifulSoup(document["text"], "lxml")

        # fragments are how APRA document web pages are structured
        fragments = soup.find_all(class_="fragment")

        current_section = None
        section_text = []

        for fragment in fragments:
            h1 = fragment.find("h1")
            if h1:
                h1.extract()

            header = fragment.find("h2")

            if header:
                if section_text:
                    results.append(
                        {
                            **document,
                            "section": current_section,
                            "text": " ".join(section_text),
                        }
                    )

                current_section = header.get_text(strip=True)
                header.extract()
                section_text = []

            text = fragment.get_text(" ", strip=True)

            if text:
                section_text.append(text)

        if section_text:
            results.append(
                {
                    **document,
                    "section": current_section,
                    "text": " ".join(section_text),
                }
            )

    return results


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
    else:
        documents = fetch_documents(MANIFEST_PATH)
        print(f"Fetched {len(documents)} documents")

        save_documents(documents, DOCUMENTS_PATH)
        print(f"Saved {len(documents)} to {DOCUMENTS_PATH}")

    documents = normalise_documents(documents)
    print(f"Normalised documents into {len(documents)} items")

    for key, value in documents[0].items():
        print(f"{key}: {value}")

    # chunks = chunk_documents(documents)
    # print(f"Created {len(chunks)} chunks")

    # save_chunks(chunks, CHUNKS_PATH)
    # print(f"Saved chunks to {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
