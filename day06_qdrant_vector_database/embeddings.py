import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI()

EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small",
)


def create_embedding(
    text: str,
) -> list[float]:

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )

    return response.data[0].embedding


if __name__ == "__main__":

    test_text = "Vector databases store embeddings."

    embedding = create_embedding(
        test_text
    )

    print(
        f"Embedding dimensions: "
        f"{len(embedding)}"
    )