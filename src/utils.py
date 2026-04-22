import json
import logging
from pathlib import Path

from rich.logging import RichHandler

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger(__name__)


def load_jsonl(jsonl_path: Path) -> list[dict]:
    logger.info("Loading records from %s", jsonl_path)
    records = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    logger.info("Loaded %d records from %s", len(records), jsonl_path)
    return records
