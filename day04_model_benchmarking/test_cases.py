TEST_CASES = [
    {
        "id": "rag_explanation",
        "category": "technical_explanation",

        "prompt": """
Explain Retrieval-Augmented Generation.

Requirements:
- Exactly 3 bullet points
- Mention embeddings
- Mention vector database
- Mention LLM
""",

        "required_keywords": [
            "embedding",
            "vector",
            "LLM",
        ],

        "max_words": 120,
    },

    {
        "id": "async_explanation",
        "category": "conciseness",

        "prompt": """
Explain why async programming is useful for LLM APIs.

Requirements:
- Maximum 2 sentences
- Mention concurrency or concurrent requests
- Do not include code
""",

        "required_keywords": [
            "concurr",
        ],

        "max_sentences": 2,
    },

    {
        "id": "http_429",
        "category": "classification",

        "prompt": """
An external LLM API returns HTTP 429.

Choose exactly one label:

RETRY
DO_NOT_RETRY

Return only the label.
""",

        "exact_answer": "RETRY",
    },

    {
        "id": "token_cost_reasoning",
        "category": "reasoning",

        "prompt": """
An AI service processes 100 requests.

Each request uses:
- 1,000 input tokens
- 500 output tokens

How many total tokens were processed?

Return only the integer.
""",

        "exact_answer": "150000",
    },

    {
        "id": "resume_extraction",
        "category": "structured_output",

        "prompt": """
Candidate resume:

Name: Alex Morgan
Skills: Python, FastAPI, SQL, Docker
Experience: 3 years

Return valid JSON only using exactly these keys:

{
    "name": "...",
    "skills": [],
    "experience_years": 0
}

Do not include Markdown.
""",

        "expected_json": {
            "name": "Alex Morgan",
            "skills": [
                "Python",
                "FastAPI",
                "SQL",
                "Docker",
            ],
            "experience_years": 3,
        },
    },

    {
        "id": "format_control",
        "category": "instruction_following",

        "prompt": """
List three advantages of vector databases.

Rules:
- Exactly 3 lines
- Each line must begin with BENEFIT:
- No introduction
- No conclusion
""",

        "required_prefix": "BENEFIT:",
        "expected_line_count": 3,
    },

    {
        "id": "rag_vs_finetuning",
        "category": "technical_reasoning",

        "prompt": """
A company has internal policy documents that change every week.

Should the company primarily use:

A) Fine-tuning
B) RAG

Return:
1. The letter
2. One sentence explaining why
""",

        "required_keywords": [
            "B",
            "RAG",
        ],
    },

    {
        "id": "api_auth_failure",
        "category": "production_reasoning",

        "prompt": """
An LLM API request fails because the API key is invalid.

Should the application retry the same request with exponential backoff?

Answer exactly:

YES

or

NO
""",

        "exact_answer": "NO",
    },
]