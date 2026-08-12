from qdrant_client.models import PointStruct

from embeddings import create_embedding
from qdrant_store import (
    COLLECTION_NAME,
    client,
    create_collection,
)


DOCUMENTS = [
    {
        "id": 1,
        "text": (
            "Python is a popular programming language "
            "used for backend development and data science."
        ),
        "category": "programming",
        "source": "programming_notes",
    },
    {
        "id": 2,
        "text": (
            "Football is a team sport where players "
            "compete to score goals."
        ),
        "category": "sports",
        "source": "sports_notes",
    },
    {
        "id": 3,
        "text": (
            "Italian cooking commonly uses pasta, "
            "tomatoes, cheese, and olive oil."
        ),
        "category": "cooking",
        "source": "cooking_notes",
    },
    {
        "id": 4,
        "text": (
            "Machine learning allows computer systems "
            "to learn patterns from data."
        ),
        "category": "machine_learning",
        "source": "ml_notes",
    },
    {
        "id": 5,
        "text": (
            "Soccer players pass and control the ball "
            "while attempting to score against an opponent."
        ),
        "category": "sports",
        "source": "sports_notes",
    },
    {
        "id": 6,
        "text": (
            "Vector databases store embeddings and "
            "support efficient similarity search."
        ),
        "category": "databases",
        "source": "database_notes",
    },
]


def ingest_documents():

    create_collection()

    points = []

    for document in DOCUMENTS:

        print(
            f"Embedding document "
            f"{document['id']}..."
        )

        embedding = create_embedding(
            document["text"]
        )

        point = PointStruct(
            id=document["id"],
            vector=embedding,
            payload={
                "text": document["text"],
                "category": document["category"],
                "source": document["source"],
            },
        )

        points.append(point)

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print(
        f"\nInserted "
        f"{len(points)} points into Qdrant."
    )

    collection_info = client.get_collection(
        COLLECTION_NAME
    )

    print(
        f"Total points stored: "
        f"{collection_info.points_count}"
    )


if __name__ == "__main__":
    ingest_documents()