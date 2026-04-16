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
        chunk = {
            "id": id,
            "title": metadata["title"],
            "section": metadata["section"],
            "text": document,
            "distance": distance,
        }
        chunks.append(chunk)

    return chunks


def main() -> None:
    openai_client = OpenAI()

    chroma_client = chromadb.PersistentClient(CHROMA_DIR)
    collection = chroma_client.get_collection(COLLECTION_NAME)

    query = "What are the rules around replacing board members?"
    chunks = retrieve_chunks(query, collection, openai_client)

    for chunk in chunks:
        print(f"\nID: {chunk['id']}")
        print(f"Title: {chunk['title']}")
        print(f"Section: {chunk['section']}")
        print(f"Distance: {chunk['distance']}")
        print(f"Text: {chunk['text']}")


if __name__ == "__main__":
    main()
