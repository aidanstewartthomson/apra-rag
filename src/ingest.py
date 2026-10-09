import hashlib
import logging
import re
from urllib.parse import urlparse
from xml.etree import ElementTree

import pysbd
import requests
import tiktoken
from bs4 import BeautifulSoup
from config import (
    CHUNK_MAX_TOKENS,
    CHUNK_MIN_TOKENS,
    CHUNKS_PATH,
    DOCUMENTS_PATH,
    EMBEDDING_MODEL,
    SITEMAP_URL,
)
from utils import configure_logging, load_jsonl, save_jsonl

logger = logging.getLogger(__name__)

SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

DOC_TYPE_BY_PREFIX = {
    "standard": "Prudential standard",
    "ppg": "Prudential practice guide",
    "letter": "Letter",
    "faq": "Frequently asked questions",
    "information-paper": "Information paper",
}

# e.g. SPS 160, CPS 230, 3PS 001, CPG 229
CODE_RE = re.compile(r"\b([123]?[A-Z]{2,3})\s+(\d{3}[A-Z]?)\b")


def load_sitemap(sitemap_url: str = SITEMAP_URL) -> list[dict]:
    response = requests.get(sitemap_url)
    response.raise_for_status()

    root = ElementTree.fromstring(response.content)
    entries = []

    for loc in root.findall("sm:url/sm:loc", SITEMAP_NS):
        url = (loc.text or "").strip()
        if not url:
            continue

        prefix = urlparse(url).path.strip("/").split("/", 1)[0]
        doc_type = DOC_TYPE_BY_PREFIX.get(prefix)
        if not doc_type:
            continue

        entries.append({"url": url, "doc_type": doc_type})

    logger.info("Loaded %d document URLs from sitemap", len(entries))
    return entries


def extract_title(html: str) -> str | None:
    soup = BeautifulSoup(html, "lxml")
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(" ", strip=True)
        if title:
            return title

    if soup.title and soup.title.string:
        return soup.title.string.split("|", 1)[0].strip()

    return None


def extract_code(title: str | None) -> str | None:
    if not title:
        return None

    match = CODE_RE.search(title)
    if not match:
        return None

    return f"{match.group(1)} {match.group(2)}"


def fetch_documents(sitemap_url: str = SITEMAP_URL) -> list[dict]:
    rows = load_sitemap(sitemap_url)
    documents = []
    total = len(rows)

    logger.info("Fetching %d documents", total)

    for i, row in enumerate(rows, start=1):
        url = row["url"]

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            logger.warning("Skipping %s (HTTP %s)", url, status)
            continue
        except requests.RequestException:
            logger.exception("Failed to fetch document from %s", url)
            continue

        title = extract_title(response.text)

        documents.append(
            {
                "url": url,
                "title": title,
                "code": extract_code(title),
                "doc_type": row["doc_type"],
                "text": response.text,
            }
        )

        if i % 25 == 0 or i == total:
            logger.info("Fetched %d/%d URLs", i, total)

    logger.info("Fetched %d/%d documents", len(documents), total)
    return documents


def normalise_documents(documents: list[dict]) -> list[dict]:
    logger.info("Normalising %d documents", len(documents))
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

    logger.info("Normalised %d documents into %d sections", len(documents), len(results))
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
    documents: list[dict],
    max_tokens: int = CHUNK_MAX_TOKENS,
    min_tokens: int = CHUNK_MIN_TOKENS,
) -> list[dict]:
    logger.info(
        "Chunking %d sections (max_tokens=%d, min_tokens=%d)",
        len(documents),
        max_tokens,
        min_tokens,
    )

    segmenter = pysbd.Segmenter(clean=True)
    encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)

    chunks = []
    total = len(documents)

    for i, document in enumerate(documents):
        sentences = segmenter.segment(document["text"])
        chunk_texts = chunk_sentences(sentences, encoding, max_tokens)

        for j, text in enumerate(chunk_texts):
            token_count = len(encoding.encode(text))
            if token_count < min_tokens:
                continue

            key = f"{document['url']}{i}{j}"
            chunk_id = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

            chunks.append(
                {
                    "id": chunk_id,
                    **document,
                    "text": text,
                }
            )

        done = i + 1
        if done % 250 == 0 or done == total:
            logger.info("Chunked %d/%d sections (%d chunks)", done, total, len(chunks))

    return chunks


def main() -> None:
    configure_logging()
    logger.info("Starting ingestion pipeline")

    if DOCUMENTS_PATH.exists():
        documents = load_jsonl(DOCUMENTS_PATH)
    else:
        documents = fetch_documents()
        save_jsonl(documents, DOCUMENTS_PATH)

    documents = normalise_documents(documents)
    chunks = chunk_documents(documents)
    save_jsonl(chunks, CHUNKS_PATH)

    logger.info("Ingestion pipeline complete")


if __name__ == "__main__":
    main()
