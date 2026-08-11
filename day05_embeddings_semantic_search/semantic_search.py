from embeddings import create_embedding
from similarity import cosine_similarity


DOCUMENTS = [
    {
        "id": 1,
        "text": (
            "Python is a popular programming language "
            "used for backend development and data science."
        ),
    },
    {
        "id": 2,
        "text": (
            "Football is a team sport where players "
            "compete to score goals."
        ),
    },
    {
        "id": 3,
        "text": (
            "Italian cooking commonly uses ingredients "
            "such as pasta, tomatoes, cheese, and olive oil."
        ),
    },
    {
        "id": 4,
        "text": (
            "Machine learning allows computer systems "
            "to learn patterns from data."
        ),
    },
    {
        "id": 5,
        "text": (
            "Soccer players pass and control the ball "
            "while attempting to score against the opponent."
        ),
    },
    {
        "id": 6,
        "text": (
            "Vector databases store embeddings and support "
            "efficient similarity search."
        ),
    },
]


def create_document_embeddings(
    documents: list[dict],
) -> list[dict]:

    embedded_documents = []

    for document in documents:

        embedding = create_embedding(
            document["text"]
        )

        embedded_documents.append(
            {
                "id": document["id"],
                "text": document["text"],
                "embedding": embedding,
            }
        )

    return embedded_documents


def semantic_search(
    query: str,
    documents: list[dict],
    top_k: int = 3,
) -> list[dict]:

    query_embedding = create_embedding(
        query
    )

    results = []

    for document in documents:

        similarity = cosine_similarity(
            query_embedding,
            document["embedding"],
        )

        results.append(
            {
                "id": document["id"],
                "text": document["text"],
                "score": similarity,
            }
        )

    results.sort(
        key=lambda result: result["score"],
        reverse=True,
    )

    return results[:top_k]


if __name__ == "__main__":

    print(
        "Creating document embeddings..."
    )

    embedded_documents = (
        create_document_embeddings(
            DOCUMENTS
        )
    )

    query = (
        "What sport involves players "
        "trying to score goals with a ball?"
    )

    print(
        f"\nQuery: {query}"
    )

    results = semantic_search(
        query=query,
        documents=embedded_documents,
        top_k=3,
    )

    print(
        "\nTop semantic search results:"
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"\nRank #{rank}"
        )

        print(
            f"Document ID: "
            f"{result['id']}"
        )

        print(
            f"Similarity: "
            f"{result['score']:.4f}"
        )

        print(
            f"Text: "
            f"{result['text']}"
        )