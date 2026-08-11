import numpy as np

from embeddings import create_embedding


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:

    a = np.array(vector_a)
    b = np.array(vector_b)

    dot_product = np.dot(a, b)

    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)

    similarity = (
        dot_product
        / (magnitude_a * magnitude_b)
    )

    return float(similarity)


if __name__ == "__main__":

    sentence_a = "I love playing football."

    sentence_b = (
        "Soccer is my favorite sport."
    )

    sentence_c = (
        "Python is used for machine learning."
    )

    sentence_d = (
        "I enjoy watching football matches."
    )

    print(
        "Creating embeddings..."
    )

    embedding_a = create_embedding(
        sentence_a
    )

    embedding_b = create_embedding(
        sentence_b
    )

    embedding_c = create_embedding(
        sentence_c
    )

    embedding_d = create_embedding(
        sentence_d
    )

    similarity_ab = cosine_similarity(
        embedding_a,
        embedding_b,
    )

    similarity_ac = cosine_similarity(
        embedding_a,
        embedding_c,
    )

    similarity_ad = cosine_similarity(
        embedding_a,
        embedding_d,
    )

    print(
        f"\nA: {sentence_a}"
    )

    print(
        f"\nB: {sentence_b}"
    )

    print(
        f"Similarity A ↔ B: "
        f"{similarity_ab:.4f}"
    )

    print(
        f"\nC: {sentence_c}"
    )

    print(
        f"Similarity A ↔ C: "
        f"{similarity_ac:.4f}"
    )

    print(
        f"\nD: {sentence_d}"
    )

    print(
        f"Similarity A ↔ D: "
        f"{similarity_ad:.4f}"
    )