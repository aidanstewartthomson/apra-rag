import json
import logging
from pathlib import Path

from rich.logging import RichHandler

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
# remove annoying openai info logs
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def load_jsonl(jsonl_path: Path) -> list[dict]:
    logger.info("Loading records from %s", jsonl_path)
    records = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    logger.info("Loaded %d records from %s", len(records), jsonl_path)
    return records


def save_jsonl(records: list[dict], jsonl_path: Path) -> None:
    logger.info("Saving %d records to %s", len(records), jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Saved %d records to %s", len(records), jsonl_path)
