import chromadb
from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, TOP_K
from openai import OpenAI


def embed_query(query: str, client: OpenAI) -> list[float]:
    response = client.embeddings.create(input=query, model=EMBEDDING_MODEL)
    return response.data[0].embedding


def retrieve_chunks(
    query: str, collection: chromadb.Collection, client: OpenAI, n_results: int = TOP_K
):
    query_embedding = embed_query(query, client)
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

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
