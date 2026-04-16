from config import EMBEDDING_MODEL
from openai import OpenAI


def embed_query(query: str, client: OpenAI) -> list[float]:
    response = client.embeddings.create(input=query, model=EMBEDDING_MODEL)
    return response.data[0].embedding


def main() -> None:
    client = OpenAI()
    query = "What are the rules around replacing board members?"

    query_embedding = embed_query(query, client)
    print(query_embedding)


if __name__ == "__main__":
    main()
