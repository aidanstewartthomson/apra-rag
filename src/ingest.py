import logging
import re
from pathlib import Path
from urllib.parse import urlparse
from xml.etree import ElementTree

import requests
from bs4 import BeautifulSoup
from config import DOCUMENTS_PATH, HTML_DIR, SITEMAP_URL
from utils import configure_logging, save_jsonl

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


def html_path_for_url(url: str, html_dir: Path = HTML_DIR) -> Path:
    parts = urlparse(url).path.strip("/").split("/")
    if not parts or not parts[0]:
        raise ValueError(f"URL has no path: {url}")

    doc_type, *rest = parts
    if not rest:
        return html_dir / doc_type / f"{doc_type}.html"

    return html_dir / doc_type / f"{'_'.join(rest)}.html"


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
    HTML_DIR.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(rows, start=1):
        url = row["url"]
        html_path = html_path_for_url(url)

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

        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(response.text, encoding="utf-8")

        title = extract_title(response.text)

        documents.append(
            {
                "url": url,
                "title": title,
                "code": extract_code(title),
                "doc_type": row["doc_type"],
                "path": str(html_path.relative_to(HTML_DIR)),
            }
        )

        if i % 25 == 0 or i == total:
            logger.info("Fetched %d/%d URLs", i, total)

    logger.info("Fetched %d/%d documents", len(documents), total)
    return documents


def main() -> None:
    configure_logging()
    logger.info("Starting ingestion pipeline")

    if DOCUMENTS_PATH.exists():
        logger.info("Documents already exist at %s, skipping fetch", DOCUMENTS_PATH)
        return

    documents = fetch_documents()
    save_jsonl(documents, DOCUMENTS_PATH)

    logger.info("Ingestion pipeline complete")


if __name__ == "__main__":
    main()
