import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small",
)

def create_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model = EMBEDDING_MODEL,
        input = text,
    )

    return response.data[0].embedding

if __name__ == "__main__":
    text = "I love playing football."
    embedding = create_embedding(text)
    print(
        f"Embedding dimensions: "
        f"{len(embedding)}"  
    )
    print(
        f"First 10 values: "
        f"{embedding[:10]}"
    )