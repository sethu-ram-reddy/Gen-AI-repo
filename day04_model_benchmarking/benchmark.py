import json
import time

from dotenv import load_dotenv
from openai import OpenAI

from models import MODELS
from test_cases import TEST_CASES


load_dotenv()

client = OpenAI()


def calculate_cost(
    model_config: dict,
    input_tokens: int,
    output_tokens: int,
) -> float:

    input_cost = (
        input_tokens / 1_000_000
    ) * model_config["input_price_per_million"]

    output_cost = (
        output_tokens / 1_000_000
    ) * model_config["output_price_per_million"]

    return input_cost + output_cost


def count_sentences(text: str) -> int:

    cleaned = text.strip()

    if not cleaned:
        return 0

    sentence_endings = (
        cleaned.count(".")
        + cleaned.count("!")
        + cleaned.count("?")
    )

    return max(
        1,
        sentence_endings,
    )


def normalize_answer(text: str) -> str:

    return (
        text
        .strip()
        .replace("`", "")
        .replace('"', "")
        .replace("'", "")
        .strip()
    )


def evaluate_response(
    test_case: dict,
    response_text: str,
) -> tuple[bool, list[str]]:

    checks = []

    text = response_text.strip()
    lower_text = text.lower()

    if "exact_answer" in test_case:

        expected = normalize_answer(
            test_case["exact_answer"]
        ).lower()

        actual = normalize_answer(
            text
        ).lower()

        passed = actual == expected

        checks.append(
            f"Exact answer: {passed}"
        )

        if not passed:
            return False, checks

    if "required_keywords" in test_case:

        keyword_results = []

        for keyword in test_case["required_keywords"]:

            found = (
                keyword.lower()
                in lower_text
            )

            keyword_results.append(
                found
            )

            checks.append(
                f"Keyword '{keyword}': {found}"
            )

        if not all(keyword_results):
            return False, checks

    if "max_words" in test_case:

        word_count = len(
            text.split()
        )

        passed = (
            word_count
            <= test_case["max_words"]
        )

        checks.append(
            f"Word count "
            f"{word_count}/"
            f"{test_case['max_words']}: "
            f"{passed}"
        )

        if not passed:
            return False, checks

    if "max_sentences" in test_case:

        sentence_count = (
            count_sentences(text)
        )

        passed = (
            sentence_count
            <= test_case["max_sentences"]
        )

        checks.append(
            f"Sentence count "
            f"{sentence_count}/"
            f"{test_case['max_sentences']}: "
            f"{passed}"
        )

        if not passed:
            return False, checks

    if "expected_line_count" in test_case:

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        expected_count = (
            test_case["expected_line_count"]
        )

        passed = (
            len(lines)
            == expected_count
        )

        checks.append(
            f"Line count "
            f"{len(lines)}/"
            f"{expected_count}: "
            f"{passed}"
        )

        if not passed:
            return False, checks

    if "required_prefix" in test_case:

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        prefix = (
            test_case["required_prefix"]
        )

        passed = (
            len(lines) > 0
            and all(
                line.startswith(prefix)
                for line in lines
            )
        )

        checks.append(
            f"Required prefix "
            f"'{prefix}': {passed}"
        )

        if not passed:
            return False, checks

    if "expected_json" in test_case:

        try:
            parsed = json.loads(text)

        except json.JSONDecodeError:

            checks.append(
                "Valid JSON: False"
            )

            return False, checks

        expected = (
            test_case["expected_json"]
        )

        passed = (
            parsed == expected
        )

        checks.append(
            f"Valid expected JSON: {passed}"
        )

        if not passed:
            return False, checks

    return True, checks


def run_single_test(
    model_id: str,
    model_config: dict,
    test_case: dict,
) -> dict:

    print(
        f"\nRunning "
        f"{model_config['name']} "
        f"→ {test_case['id']}"
    )

    start_time = (
        time.perf_counter()
    )

    try:

        response = client.responses.create(
            model=model_id,
            input=test_case["prompt"],
        )

        latency = (
            time.perf_counter()
            - start_time
        )

        response_text = (
            response.output_text
        )

        input_tokens = (
            response.usage.input_tokens
        )

        output_tokens = (
            response.usage.output_tokens
        )

        total_tokens = (
            response.usage.total_tokens
        )

        cost = calculate_cost(
            model_config=model_config,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        passed, checks = (
            evaluate_response(
                test_case=test_case,
                response_text=response_text,
            )
        )

        print(
            f"PASS: {passed}"
        )

        print(
            f"Latency: {latency:.2f}s"
        )

        print(
            f"Tokens: {total_tokens}"
        )

        print(
            f"Cost: ${cost:.6f}"
        )

        return {
            "model_id": model_id,
            "model_name": model_config["name"],
            "test_id": test_case["id"],
            "category": test_case["category"],
            "success": True,
            "passed": passed,
            "response": response_text,
            "checks": checks,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "latency_seconds": round(
                latency,
                3,
            ),
            "cost_usd": round(
                cost,
                8,
            ),
            "error": None,
        }

    except Exception as error:

        latency = (
            time.perf_counter()
            - start_time
        )

        print(
            f"ERROR: "
            f"{type(error).__name__}"
        )

        return {
            "model_id": model_id,
            "model_name": model_config["name"],
            "test_id": test_case["id"],
            "category": test_case["category"],
            "success": False,
            "passed": False,
            "response": None,
            "checks": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "latency_seconds": round(
                latency,
                3,
            ),
            "cost_usd": 0,
            "error": (
                f"{type(error).__name__}: "
                f"{str(error)}"
            ),
        }


def run_benchmark() -> list[dict]:

    results = []

    print(
        "\n========================================"
    )
    print(
        "DAY 4 - MODEL BENCHMARK"
    )
    print(
        "========================================"
    )

    for model_id, model_config in (
        MODELS.items()
    ):

        print(
            "\n----------------------------------------"
        )

        print(
            f"MODEL: "
            f"{model_config['name']}"
        )

        print(
            "----------------------------------------"
        )

        for test_case in TEST_CASES:

            result = run_single_test(
                model_id=model_id,
                model_config=model_config,
                test_case=test_case,
            )

            results.append(
                result
            )

    return results


def generate_summary(
    results: list[dict],
) -> list[dict]:

    summaries = []

    for model_id, model_config in (
        MODELS.items()
    ):

        model_results = [
            result
            for result in results
            if result["model_id"] == model_id
        ]

        total_tests = len(
            model_results
        )

        successful_api_calls = sum(
            1
            for result in model_results
            if result["success"]
        )

        passed_tests = sum(
            1
            for result in model_results
            if result["passed"]
        )

        total_latency = sum(
            result["latency_seconds"]
            for result in model_results
        )

        total_tokens = sum(
            result["total_tokens"]
            for result in model_results
        )

        total_cost = sum(
            result["cost_usd"]
            for result in model_results
        )

        average_latency = (
            total_latency / total_tests
            if total_tests
            else 0
        )

        pass_rate = (
            (
                passed_tests
                / total_tests
            )
            * 100
            if total_tests
            else 0
        )

        api_success_rate = (
            (
                successful_api_calls
                / total_tests
            )
            * 100
            if total_tests
            else 0
        )

        summaries.append(
            {
                "model_id": model_id,
                "model_name": model_config["name"],
                "tests": total_tests,
                "passed_tests": passed_tests,
                "pass_rate_percent": round(
                    pass_rate,
                    2,
                ),
                "api_success_rate_percent": round(
                    api_success_rate,
                    2,
                ),
                "average_latency_seconds": round(
                    average_latency,
                    3,
                ),
                "total_tokens": total_tokens,
                "total_cost_usd": round(
                    total_cost,
                    8,
                ),
            }
        )

    return summaries


def generate_category_analysis(
    results: list[dict],
) -> list[dict]:

    analysis = []

    categories = sorted(
        set(
            result["category"]
            for result in results
        )
    )

    for category in categories:

        for model_id, model_config in (
            MODELS.items()
        ):

            category_results = [
                result
                for result in results
                if (
                    result["model_id"] == model_id
                    and result["category"] == category
                )
            ]

            if not category_results:
                continue

            total = len(
                category_results
            )

            passed = sum(
                1
                for result in category_results
                if result["passed"]
            )

            successful_calls = sum(
                1
                for result in category_results
                if result["success"]
            )

            average_latency = (
                sum(
                    result["latency_seconds"]
                    for result
                    in category_results
                )
                / total
            )

            total_tokens = sum(
                result["total_tokens"]
                for result
                in category_results
            )

            total_cost = sum(
                result["cost_usd"]
                for result
                in category_results
            )

            pass_rate = (
                passed / total
            ) * 100

            api_success_rate = (
                successful_calls / total
            ) * 100

            analysis.append(
                {
                    "category": category,
                    "model_id": model_id,
                    "model_name": model_config["name"],
                    "tests": total,
                    "passed_tests": passed,
                    "pass_rate_percent": round(
                        pass_rate,
                        2,
                    ),
                    "api_success_rate_percent": round(
                        api_success_rate,
                        2,
                    ),
                    "average_latency_seconds": round(
                        average_latency,
                        3,
                    ),
                    "total_tokens": total_tokens,
                    "total_cost_usd": round(
                        total_cost,
                        8,
                    ),
                }
            )

    return analysis


def print_leaderboard(
    summaries: list[dict],
):

    print(
        "\n\n========================================"
    )
    print(
        "FINAL MODEL LEADERBOARD"
    )
    print(
        "========================================"
    )

    ranked = sorted(
        summaries,
        key=lambda item: (
            -item["pass_rate_percent"],
            item["total_cost_usd"],
            item["average_latency_seconds"],
        ),
    )

    for rank, summary in enumerate(
        ranked,
        start=1,
    ):

        print(
            f"\n#{rank} "
            f"{summary['model_name']}"
        )

        print(
            f"Passed: "
            f"{summary['passed_tests']}/"
            f"{summary['tests']}"
        )

        print(
            f"Pass Rate: "
            f"{summary['pass_rate_percent']}%"
        )

        print(
            f"API Success: "
            f"{summary['api_success_rate_percent']}%"
        )

        print(
            f"Average Latency: "
            f"{summary['average_latency_seconds']}s"
        )

        print(
            f"Total Tokens: "
            f"{summary['total_tokens']}"
        )

        print(
            f"Total Cost: "
            f"${summary['total_cost_usd']:.6f}"
        )


def print_category_analysis(
    analysis: list[dict],
):

    print(
        "\n\n========================================"
    )
    print(
        "CATEGORY-LEVEL ANALYSIS"
    )
    print(
        "========================================"
    )

    categories = sorted(
        set(
            item["category"]
            for item in analysis
        )
    )

    for category in categories:

        print(
            f"\n{category.upper()}"
        )

        category_items = [
            item
            for item in analysis
            if item["category"] == category
        ]

        category_items = sorted(
            category_items,
            key=lambda item: (
                -item["pass_rate_percent"],
                item["total_cost_usd"],
                item[
                    "average_latency_seconds"
                ],
            ),
        )

        for item in category_items:

            print(
                f"{item['model_name']}: "
                f"{item['passed_tests']}/"
                f"{item['tests']} passed | "
                f"{item['pass_rate_percent']}% | "
                f"{item['average_latency_seconds']}s | "
                f"{item['total_tokens']} tokens | "
                f"${item['total_cost_usd']:.6f}"
            )


def save_report(
    results: list[dict],
    summaries: list[dict],
    category_analysis: list[dict],
):

    report = {
        "results": results,
        "summary": summaries,
        "category_analysis": category_analysis,
    }

    with open(
        "benchmark_results.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )

    print(
        "\nReport saved to "
        "benchmark_results.json"
    )


if __name__ == "__main__":

    benchmark_results = (
        run_benchmark()
    )

    benchmark_summary = (
        generate_summary(
            benchmark_results
        )
    )

    category_analysis = (
        generate_category_analysis(
            benchmark_results
        )
    )

    print_leaderboard(
        benchmark_summary
    )

    print_category_analysis(
        category_analysis
    )

    save_report(
        benchmark_results,
        benchmark_summary,
        category_analysis,
    )