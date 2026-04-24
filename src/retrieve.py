from chromadb import Collection

from config import TOP_K


def retrieve_chunks(query: str, collection: Collection, n_results: int = TOP_K):
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
