import json

from embeddings import create_embedding
from similarity import cosine_similarity


def load_vector_store(
    filename: str = "vector_store.json",
) -> list[dict]:

    with open(
        filename,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def semantic_search(
    query: str,
    vector_store: list[dict],
    top_k: int = 3,
) -> list[dict]:

    query_embedding = create_embedding(
        query
    )

    results = []

    for document in vector_store:

        score = cosine_similarity(
            query_embedding,
            document["embedding"],
        )

        results.append(
            {
                "id": document["id"],
                "text": document["text"],
                "category": document["category"],
                "score": score,
            }
        )

    results.sort(
        key=lambda result: result["score"],
        reverse=True,
    )

    return results[:top_k]


if __name__ == "__main__":

    vector_store = load_vector_store()

    query = input(
        "Enter your search query: "
    )

    results = semantic_search(
        query=query,
        vector_store=vector_store,
        top_k=3,
    )

    print(
        "\nTop results:"
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"\nRank #{rank}"
        )

        print(
            f"Score: "
            f"{result['score']:.4f}"
        )

        print(
            f"Category: "
            f"{result['category']}"
        )

        print(
            f"Text: "
            f"{result['text']}"
        )