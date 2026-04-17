import chromadb
from openai import OpenAI
from config import CHROMA_DIR, COLLECTION_NAME, GENERATION_MODEL, SYSTEM_INSTRUCTIONS
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
    prompt = "\n".join(["Context:", "", context, "", "Question:", query])
    return prompt


def generate_answer(prompt: str, client: OpenAI) -> str:
    response = client.responses.create(
        model=GENERATION_MODEL,
        instructions=SYSTEM_INSTRUCTIONS,
        input=prompt,
    )
    return response.output_text


def main() -> None:
    chroma_client = chromadb.PersistentClient(CHROMA_DIR)
    collection = chroma_client.get_collection(COLLECTION_NAME)

    openai_client = OpenAI()

    while True:
        query = input("\nQuery: ")
        chunks = retrieve_chunks(query, collection, openai_client)

        context = build_context(chunks)
        prompt = build_prompt(query, context)

        answer = generate_answer(prompt, openai_client)

        print("\nAnswer:")
        print(answer)


if __name__ == "__main__":
    main()
