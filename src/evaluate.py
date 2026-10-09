import argparse
import logging
import sys
from typing import Callable

from bm25s import BM25

from config import (
    BM25_DIR,
    QUERIES_PATH,
    SEARCH_METHOD,
    SPARSE_INDEX_NAME,
    TOP_K,
)
from retrieve import hybrid_search, keyword_search, semantic_search
from utils import configure_logging, get_chroma_collection, load_jsonl

logger = logging.getLogger(__name__)


def evaluate_queries(
    queries: list[dict],
    search_function: Callable,
    search_method: str,
    n_results: int,
) -> dict:
    logger.info(
        "Evaluating %d queries (method=%s, top_k=%d)",
        len(queries),
        search_method,
        n_results,
    )

    reciprocal_ranks = []
    hits = 0
    total = len(queries)

    for i, query in enumerate(queries, start=1):
        chunks = search_function(query["text"], n_results)

        reciprocal_rank = 0
        for rank, chunk in enumerate(chunks, start=1):
            if chunk["id"] == query["chunk_id"]:
                reciprocal_rank = 1 / rank
                hits += 1
                break

        reciprocal_ranks.append(reciprocal_rank)

        if i % 25 == 0 or i == total:
            logger.info("Evaluated %d/%d queries", i, total)

    recall = hits / total
    mrr = sum(reciprocal_ranks) / total

    logger.info("recall=%.3f mrr=%.3f hits=%d/%d", recall, mrr, hits, total)

    return {
        "recall": recall,
        "mrr": mrr,
        "top_k": n_results,
        "hits": hits,
        "total": total,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval on a fixed query set"
    )
    parser.add_argument(
        "--method",
        choices=["keyword", "semantic", "hybrid"],
        default=SEARCH_METHOD,
        help=f"retrieval method (default: {SEARCH_METHOD})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"number of results to retrieve (default: {TOP_K})",
    )

    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    if args.top_k < 1:
        logger.error("--top-k must be >= 1")
        sys.exit(1)

    if not QUERIES_PATH.exists():
        logger.error(
            "No query set at %s. Create one with: uv run --env-file .env make-queries",
            QUERIES_PATH,
        )
        sys.exit(1)

    queries = load_jsonl(QUERIES_PATH)

    retriever = BM25.load(BM25_DIR / SPARSE_INDEX_NAME, load_corpus=True)
    collection = get_chroma_collection()

    match args.method:
        case "keyword":
            search_fn = lambda q, k: keyword_search(q, retriever, k)
        case "semantic":
            search_fn = lambda q, k: semantic_search(q, collection, k)
        case "hybrid":
            search_fn = lambda q, k: hybrid_search(q, retriever, collection, k)

    evaluate_queries(queries, search_fn, args.method, args.top_k)


if __name__ == "__main__":
    main()
