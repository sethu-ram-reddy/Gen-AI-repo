from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
)


COLLECTION_NAME = "day06_documents"
VECTOR_SIZE = 1536

client = QdrantClient(
    path="qdrant_data"
)


def create_collection():

    if client.collection_exists(
        collection_name=COLLECTION_NAME
    ):

        print(
            f"Collection "
            f"'{COLLECTION_NAME}' "
            f"already exists."
        )

        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )

    print(
        f"Collection "
        f"'{COLLECTION_NAME}' "
        f"created successfully."
    )


if __name__ == "__main__":

    create_collection()

    collection_info = (
        client.get_collection(
            COLLECTION_NAME
        )
    )

    print(
        f"\nCollection: "
        f"{COLLECTION_NAME}"
    )

    print(
        f"Points stored: "
        f"{collection_info.points_count}"
    )