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


def build_prompt(query: str, context: str) -> str:
    prompt = (
        "You are answering questions about APRA regulatory documents.\n\n"
        "Answer the question using only the context provided.\n"
        "Do not use prior knowledge or make assumptions.\n\n"
        "If the context does not contain enough information, say:\n"
        '"The context does not contain enough information to answer this question."\n\n'
        "When possible, cite the relevant section and title.\n\n"
        "Context:\n\n"
        f"{context}\n\n"
        "Question:\n"
        f"{query}\n\n"
        "Answer:"
    )

    return prompt


def main() -> None:
    chroma_client = chromadb.PersistentClient(CHROMA_DIR)
    collection = chroma_client.get_collection(COLLECTION_NAME)

    openai_client = OpenAI()

    query = "What are the rules around replacing board members?"
    chunks = retrieve_chunks(query, collection, openai_client)

    context = build_context(chunks)
    prompt = build_prompt(query, context)

    print(prompt)


if __name__ == "__main__":
    main()
