import time
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()


# Create FastAPI application
app = FastAPI(
    title="Day 1 - LLM API",
    description="Simple AI API built with FastAPI and OpenAI",
    version="1.0.0"
)


# Create OpenAI client
client = OpenAI()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# Request model
class ChatRequest(BaseModel):
    message: str


# Health check endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# Chat endpoint
@app.post("/chat")
def chat(request: ChatRequest):

    try:
        # Start latency timer
        start_time = time.time()

        # Call the LLM
        response = client.responses.create(
            model="gpt-5.6-luna",

            instructions="""
            You are an AI engineering tutor.

            Rules:
            - Explain concepts clearly.
            - Keep answers concise.
            - Use simple examples when useful.
            - Avoid unnecessary jargon.
            """,

            input=request.message
        )

        # Calculate latency
        latency = time.time() - start_time

        # Log request information
        logger.info(
            f"LLM request completed | "
            f"tokens={response.usage.total_tokens} | "
            f"latency={round(latency, 2)}s"
        )

        # Return response
        return {
            "response": response.output_text,

            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens
            },

            "latency_seconds": round(latency, 2)
        }

    except Exception as e:

        # Log full error internally
        logger.exception("LLM request failed")

        # Return safe error message to user
        raise HTTPException(
            status_code=500,
            detail="Failed to generate AI response."
        )