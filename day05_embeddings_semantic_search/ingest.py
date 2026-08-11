import json

from embeddings import create_embedding


DOCUMENTS = [
    {
        "id": 1,
        "text": (
            "Python is a popular programming language "
            "used for backend development and data science."
        ),
        "category": "programming",
    },
    {
        "id": 2,
        "text": (
            "Football is a team sport where players "
            "compete to score goals."
        ),
        "category": "sports",
    },
    {
        "id": 3,
        "text": (
            "Italian cooking commonly uses ingredients "
            "such as pasta, tomatoes, cheese, and olive oil."
        ),
        "category": "cooking",
    },
    {
        "id": 4,
        "text": (
            "Machine learning allows computer systems "
            "to learn patterns from data."
        ),
        "category": "machine_learning",
    },
    {
        "id": 5,
        "text": (
            "Soccer players pass and control the ball "
            "while attempting to score against the opponent."
        ),
        "category": "sports",
    },
    {
        "id": 6,
        "text": (
            "Vector databases store embeddings and support "
            "efficient similarity search."
        ),
        "category": "databases",
    },
]


def ingest_documents(
    documents: list[dict],
) -> list[dict]:

    vector_store = []

    for document in documents:

        print(
            f"Embedding document "
            f"{document['id']}..."
        )

        embedding = create_embedding(
            document["text"]
        )

        vector_store.append(
            {
                "id": document["id"],
                "text": document["text"],
                "category": document["category"],
                "embedding": embedding,
            }
        )

    return vector_store


def save_vector_store(
    vector_store: list[dict],
    filename: str = "vector_store.json",
):

    with open(
        filename,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            vector_store,
            file,
            indent=2,
        )

    print(
        f"\nSaved "
        f"{len(vector_store)} documents "
        f"to {filename}"
    )


if __name__ == "__main__":

    vector_store = ingest_documents(
        DOCUMENTS
    )

    save_vector_store(
        vector_store
    )