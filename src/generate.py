from bm25s import BM25
from openai import OpenAI

from config import (
    BM25_DIR,
    GENERATION_MODEL,
    SEARCH_METHOD,
    SPARSE_INDEX_NAME,
    SYSTEM_INSTRUCTIONS,
)
from retrieve import hybrid_search, keyword_search, semantic_search
from utils import get_chroma_collection


def build_context(chunks: list[dict]) -> str:
    blocks = []

    for i, chunk in enumerate(chunks, start=1):
        block = "\n".join(
            [
                f"Source {i}",
                f"Document: {chunk.get('code')}",
                f"Type: {chunk.get('doc_type')}",
                f"Industry: {chunk.get('industry')}",
                f"Title: {chunk.get('title')}",
                f"Section: {chunk.get('section')}",
                f"Text: {chunk.get('text')}",
            ]
        )
        blocks.append(block)

    context = "\n\n".join(blocks)
    return context


def build_prompt(query: str, context: str) -> str:
    prompt = "\n".join(["Question:", query, "", "Context:", "", context, "", "Answer:"])
    return prompt


def generate_answer(prompt: str, client: OpenAI) -> str:
    response = client.responses.create(
        model=GENERATION_MODEL,
        instructions=SYSTEM_INSTRUCTIONS,
        input=prompt,
    )
    return response.output_text


def main() -> None:
    openai_client = OpenAI()

    retriever = BM25.load(BM25_DIR / SPARSE_INDEX_NAME, load_corpus=True)
    collection = get_chroma_collection()

    while True:
        query = input("\nQuery: ")

        match SEARCH_METHOD:
            case "keyword":
                chunks = keyword_search(query, retriever)
            case "semantic":
                chunks = semantic_search(query, collection)
            case "hybrid":
                chunks = hybrid_search(query, retriever, collection)

        context = build_context(chunks)
        prompt = build_prompt(query, context)

        answer = generate_answer(prompt, openai_client)

        print(f"\n{answer}")


if __name__ == "__main__":
    main()
