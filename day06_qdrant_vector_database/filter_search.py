from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
)

from embeddings import create_embedding
from qdrant_store import (
    COLLECTION_NAME,
    client,
)


def filtered_search(
    query: str,
    category: str,
    top_k: int = 3,
):

    query_embedding = create_embedding(
        query
    )

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="category",
                    match=MatchValue(
                        value=category
                    ),
                )
            ]
        ),
        limit=top_k,
        with_payload=True,
    )

    return response.points


if __name__ == "__main__":

    query = input(
        "Enter your search query: "
    )

    category = input(
        "Enter category filter: "
    )

    results = filtered_search(
        query=query,
        category=category,
        top_k=3,
    )

    print(
        f"\nResults for category "
        f"'{category}':"
    )

    if not results:

        print(
            "\nNo matching points found."
        )

    for rank, point in enumerate(
        results,
        start=1,
    ):

        print(
            f"\nRank #{rank}"
        )

        print(
            f"Point ID: "
            f"{point.id}"
        )

        print(
            f"Score: "
            f"{point.score:.4f}"
        )

        print(
            f"Category: "
            f"{point.payload['category']}"
        )

        print(
            f"Source: "
            f"{point.payload['source']}"
        )

        print(
            f"Text: "
            f"{point.payload['text']}"
        )