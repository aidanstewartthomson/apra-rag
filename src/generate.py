import chromadb
from openai import OpenAI
from config import CHROMA_DIR, COLLECTION_NAME
from retrieve import retrieve_chunks


def build_context(chunks: list[dict]) -> str:
    blocks = []

    for i, chunk in enumerate(chunks, start=1):
        block = "\n".join(
            [
                f"Source {i}",
                f"Title: {chunk['title']}",
                f"Section: {chunk['section']}",
                f"Text: {chunk['text']}",
            ]
        )
        blocks.append(block)

    context = "\n\n".join(blocks)
    return context


def main():
    chroma_client = chromadb.PersistentClient(CHROMA_DIR)
    collection = chroma_client.get_collection(COLLECTION_NAME)

    openai_client = OpenAI()

    query = "What are the rules around replacing board members?"
    chunks = retrieve_chunks(query, collection, openai_client)

    context = build_context(chunks)
    print(context)


if __name__ == "__main__":
    main()
