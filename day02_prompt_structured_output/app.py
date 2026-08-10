import logging
import time
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

load_dotenv()

MODEL_NAME = "gpt-5.6-luna"
PROMPT_VERSION = "2.0"

client = OpenAI()

app = FastAPI(
    title="AI Resume Analyzer",
    description="Analyzes resumes using structured LLM outputs.",
    version="2.0.0",
)


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Request schema
# ---------------------------------------------------------

class ResumeRequest(BaseModel):

    resume_text: str = Field(
        min_length=50,
        description="Complete resume text to analyze",
    )

    target_role: str = Field(
        default="AI Engineer",
        min_length=2,
        description="Role against which the candidate should be evaluated",
    )

    prompt_mode: Literal["zero_shot", "few_shot"] = "zero_shot"


# ---------------------------------------------------------
# Structured response schema
# ---------------------------------------------------------

class ResumeAnalysis(BaseModel):

    candidate_name: str | None = Field(
        description=(
            "Candidate's name if explicitly present in the resume; "
            "otherwise null"
        )
    )

    professional_summary: str = Field(
        description="A concise summary of the candidate's professional profile"
    )

    skills: list[str] = Field(
        description="Technical and professional skills explicitly found in the resume"
    )

    strengths: list[str] = Field(
        description="The candidate's strongest qualifications for the target role"
    )

    skill_gaps: list[str] = Field(
        description="Important missing or weak skills for the target role"
    )

    recommended_roles: list[str] = Field(
        description="Realistic job roles matching the candidate's current profile"
    )

    overall_score: int = Field(
        ge=0,
        le=100,
        description="Readiness score for the target role from 0 to 100",
    )


# ---------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------

def build_prompt(
    target_role: str,
    prompt_mode: str,
) -> str:

    base_prompt = f"""
    You are an expert technical recruiter evaluating a candidate
    for the target role: {target_role}.

    Analyze only information explicitly present in the resume.

    Rules:
    - Do not invent skills, experience, achievements, or education.
    - Keep the professional summary concise.
    - Extract skills explicitly found in the resume.
    - Identify strengths relevant to the target role.
    - Identify meaningful gaps for the target role.
    - Recommend realistic roles based on demonstrated experience.
    - Score readiness for the target role from 0 to 100.
    - Treat missing information as unknown rather than assuming it exists.

    Scoring guideline:
    - 90-100: Exceptional readiness
    - 75-89: Strong candidate
    - 60-74: Moderate readiness
    - 40-59: Significant gaps
    - 0-39: Poor match

    If the candidate's name is unavailable, return null.
    """

    if prompt_mode == "few_shot":

        base_prompt += """

        Example:

        Candidate profile:
        Python developer with FastAPI, SQL, Docker, and AWS.
        Built a basic RAG application using embeddings and a vector database.
        No evidence of production deployment, evaluation pipelines,
        fine-tuning, agents, or MLOps.

        Target role:
        AI Engineer

        Expected evaluation behavior:
        - Recognize Python, FastAPI, and RAG as relevant strengths.
        - Do not assume production AI experience.
        - Identify evaluation, deployment, and MLOps as gaps.
        - Do not give an extremely high score simply because AI keywords exist.
        - Recommend roles appropriate to demonstrated experience.

        Example score range:
        Approximately 60-75 depending on demonstrated project depth.

        Important:
        Use this example only to understand evaluation standards.
        Do not copy its score, strengths, gaps, or recommendations
        to the actual candidate.
        """

    return base_prompt


# ---------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "AI Resume Analyzer",
        "prompt_version": PROMPT_VERSION,
        "model": MODEL_NAME,
    }


# ---------------------------------------------------------
# Resume analysis endpoint
# ---------------------------------------------------------

@app.post(
    "/analyze-resume",
    response_model=ResumeAnalysis,
)
def analyze_resume(request: ResumeRequest):

    try:

        start_time = time.time()

        # Build prompt dynamically
        instructions = build_prompt(
            target_role=request.target_role,
            prompt_mode=request.prompt_mode,
        )

        # Structured LLM call
        response = client.response.parse(
            model=MODEL_NAME,
            instructions=instructions,
            input=request.resume_text,
            text_format=ResumeAnalysis,
        )

        analysis = response.output_parsed

        if analysis is None:
            raise ValueError(
                "The model did not return parsed resume analysis."
            )

        latency = time.time() - start_time

        # Log useful metadata
        logger.info(
            "Resume analysis completed | "
            "target_role=%s | "
            "prompt_mode=%s | "
            "prompt_version=%s | "
            "score=%s | "
            "tokens=%s | "
            "latency=%.2fs",
            request.target_role,
            request.prompt_mode,
            PROMPT_VERSION,
            analysis.overall_score,
            response.usage.total_tokens,
            latency,
        )

        return analysis

    except Exception:

        logger.exception("Resume analysis failed")

        raise HTTPException(
            status_code=500,
            detail="Failed to analyze the resume.",
        )