import bm25s
from bm25s import BM25
from chromadb import Collection

from config import BM25_DIR, SPARSE_INDEX_NAME, TOP_K


def keyword_search(query: str, retriever: BM25, n_results: int = TOP_K) -> list[dict]:
    tokens = bm25s.tokenize(query)
    results, scores = retriever.retrieve(tokens, k=n_results)

    # results are batched but we only want the first query
    return [
        {**result, "score": float(score)}
        for result, score in zip(results[0], scores[0])
    ]


def retrieve_chunks(
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


def main() -> None:
    retriever = BM25.load(BM25_DIR / SPARSE_INDEX_NAME, load_corpus=True)
    query = "What capital requirements do banks have to meet under APRA?"

    results = keyword_search(query, retriever)
    print(results[:5])


if __name__ == "__main__":
    main()
