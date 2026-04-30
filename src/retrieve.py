import logging

import bm25s
from bm25s import BM25
from chromadb import Collection

from config import BM25_DIR, RRF_K, SPARSE_INDEX_NAME, TOP_K
from utils import get_chroma_collection

# silence noisy logs
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def keyword_search(query: str, retriever: BM25, n_results: int = TOP_K) -> list[dict]:
    tokens = bm25s.tokenize(query)
    results, scores = retriever.retrieve(tokens, k=n_results)

    # results are batched but we only have one query
    return [
        {**result, "score": float(score)}
        for result, score in zip(results[0], scores[0])
    ]


def semantic_search(
    query: str, collection: Collection, n_results: int = TOP_K
) -> list[dict]:
    results = collection.query(query_texts=[query], n_results=n_results)

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    chunks = []
    for id, document, metadata, distance in zip(ids, documents, metadatas, distances):
        chunks.append(
            {
                "id": id,
                **metadata,
                "text": document,
                "distance": distance,
            }
        )

    return chunks


def hybrid_search(
    query: str, retriever: BM25, collection: Collection, n_results: int = TOP_K
) -> list[dict]:
    result_sets = [
        keyword_search(query, retriever, n_results=n_results),
        semantic_search(query, collection, n_results=n_results),
    ]

    chunks = {}
    scores = {}

    for result_set in result_sets:
        for rank, result in enumerate(result_set, start=1):
            chunk_id = result["id"]
            chunks.setdefault(chunk_id, result)

            rrf_score = 1 / (RRF_K + rank)
            scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score

    ranked_ids = sorted(scores, key=scores.get, reverse=True)[:n_results]
    results = [
        {**chunks[chunk_id], "score": scores[chunk_id]} for chunk_id in ranked_ids
    ]

    return results


def main() -> None:
    retriever = BM25.load(BM25_DIR / SPARSE_INDEX_NAME, load_corpus=True)
    collection = get_chroma_collection()

    query = "What capital requirements do banks have to meet under APRA?"

    results = hybrid_search(query, retriever, collection)
    print(results)


if __name__ == "__main__":
    main()
