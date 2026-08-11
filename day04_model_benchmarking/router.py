from typing import Literal


Complexity = Literal[
    "low",
    "medium",
    "high",
]


def choose_model(
    complexity: Complexity,
) -> str:

    if complexity == "low":
        return "gpt-5.6-luna"

    if complexity == "medium":
        return "gpt-5.6-terra"

    return "gpt-5.6-sol"


def explain_routing(
    task: str,
    complexity: Complexity,
) -> dict:

    model = choose_model(
        complexity
    )

    return {
        "task": task,
        "complexity": complexity,
        "selected_model": model,
    }


if __name__ == "__main__":

    examples = [
        (
            "Extract skills from a resume",
            "low",
        ),

        (
            "Compare two RAG architectures",
            "medium",
        ),

        (
            "Design and critique a distributed "
            "multi-agent production architecture",
            "high",
        ),
    ]

    for task, complexity in examples:

        result = explain_routing(
            task=task,
            complexity=complexity,
        )

        print(
            f"\nTask: "
            f"{result['task']}"
        )

        print(
            f"Complexity: "
            f"{result['complexity']}"
        )

        print(
            f"Selected model: "
            f"{result['selected_model']}"
        )