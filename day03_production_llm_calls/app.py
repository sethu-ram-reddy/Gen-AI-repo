import asyncio
import logging
import os
import random
import time
from typing import Annotated, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import (
    AsyncOpenAI,
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
    RateLimitError,
)
from pydantic import BaseModel, Field


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

load_dotenv()


PRIMARY_MODEL_NAME = os.getenv(
    "OPENAI_MODEL",
    "gpt-5.6-luna",
)

# Optional fallback model.
#
# Example inside .env:
#
# OPENAI_FALLBACK_MODEL=your-second-model
#
# If it is not provided, fallback is disabled.
FALLBACK_MODEL_NAME = os.getenv(
    "OPENAI_FALLBACK_MODEL"
)


# Maximum time allowed for ONE model attempt.
REQUEST_TIMEOUT_SECONDS = 10.0


# Number of retries AFTER the first attempt.
#
# MAX_RETRIES = 3 means:
#
# Attempt 1
# Retry 1
# Retry 2
# Retry 3
#
# Maximum = 4 calls per model.
MAX_RETRIES = 3


# Exponential backoff configuration.
BASE_RETRY_DELAY_SECONDS = 1.0
MAX_RETRY_DELAY_SECONDS = 8.0


# Disable SDK retries because we are implementing
# our own retry system for learning purposes.
client = AsyncOpenAI(
    max_retries=0,
)


# ---------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------

app = FastAPI(
    title="Day 3 - Production LLM API",
    description=(
        "Production-oriented LLM backend with async calls, "
        "concurrency, timeouts, retries, exponential backoff, "
        "rate-limit handling, fallback models and resilient batches."
    ),
    version="1.4.0",
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
# Shared Types
# ---------------------------------------------------------

MessageText = Annotated[
    str,
    Field(
        min_length=1,
        max_length=4000,
        description="Message to send to the AI assistant",
    ),
]


# ---------------------------------------------------------
# Request Models
# ---------------------------------------------------------

class ChatRequest(BaseModel):
    message: MessageText


class BatchChatRequest(BaseModel):

    messages: list[MessageText] = Field(
        min_length=2,
        max_length=5,
        description="Two to five messages to process",
    )

    mode: Literal[
        "sequential",
        "concurrent",
    ] = "concurrent"


# ---------------------------------------------------------
# Response Models
# ---------------------------------------------------------

class ChatResponse(BaseModel):

    response: str

    model: str

    fallback_used: bool

    input_tokens: int

    output_tokens: int

    total_tokens: int

    attempts: int

    latency_seconds: float


class GeneratedAnswer(BaseModel):

    message: str

    response: str

    model: str

    fallback_used: bool

    input_tokens: int

    output_tokens: int

    total_tokens: int

    attempts: int

    latency_seconds: float


class BatchItemResult(BaseModel):

    message: str

    success: bool

    result: GeneratedAnswer | None = None

    error_type: str | None = None

    error_message: str | None = None


class BatchChatResponse(BaseModel):

    mode: Literal[
        "sequential",
        "concurrent",
    ]

    request_count: int

    successful_count: int

    failed_count: int

    total_tokens: int

    total_latency_seconds: float

    results: list[BatchItemResult]


# ---------------------------------------------------------
# Custom Model Failure
# ---------------------------------------------------------

class ModelCallFailed(Exception):

    def __init__(
        self,
        model_name: str,
        attempts: int,
        original_error: Exception,
    ):

        self.model_name = model_name

        self.attempts = attempts

        self.original_error = original_error

        super().__init__(
            f"{model_name} failed after "
            f"{attempts} attempt(s): "
            f"{type(original_error).__name__}"
        )


# ---------------------------------------------------------
# Determine Whether Error Is Retryable
# ---------------------------------------------------------

def is_retryable_error(
    error: Exception,
) -> bool:

    # Local asyncio timeout
    if isinstance(
        error,
        TimeoutError,
    ):
        return True


    # OpenAI connection / timeout errors
    if isinstance(
        error,
        (
            APIConnectionError,
            APITimeoutError,
        ),
    ):
        return True


    # Rate limit
    if isinstance(
        error,
        RateLimitError,
    ):
        return True


    # HTTP API errors
    if isinstance(
        error,
        APIStatusError,
    ):

        status_code = error.status_code

        if status_code in (
            408,
            409,
            429,
        ):
            return True

        if status_code >= 500:
            return True


    return False


# ---------------------------------------------------------
# Exponential Backoff
# ---------------------------------------------------------

def calculate_retry_delay(
    retry_number: int,
) -> float:

    exponential_delay = (
        BASE_RETRY_DELAY_SECONDS
        * (2 ** retry_number)
    )

    capped_delay = min(
        exponential_delay,
        MAX_RETRY_DELAY_SECONDS,
    )


    # Jitter prevents many requests from retrying
    # at exactly the same time.
    jitter = random.uniform(
        0,
        capped_delay * 0.25,
    )


    return capped_delay + jitter


# ---------------------------------------------------------
# Call One Model With Retries
# ---------------------------------------------------------

async def call_model_with_retries(
    model_name: str,
    message: str,
):

    for attempt_index in range(
        MAX_RETRIES + 1
    ):

        attempt_number = (
            attempt_index + 1
        )

        try:

            logger.info(
                "Starting model attempt | "
                "model=%s | "
                "attempt=%s/%s",
                model_name,
                attempt_number,
                MAX_RETRIES + 1,
            )


            async with asyncio.timeout(
                REQUEST_TIMEOUT_SECONDS
            ):

                response = (
                    await client.responses.create(

                        model=model_name,

                        instructions="""
                        You are a helpful AI engineering assistant.

                        Rules:
                        - Be technically accurate.
                        - Keep the response under three sentences.
                        - Use simple examples when helpful.
                        """,

                        input=message,
                    )
                )


            logger.info(
                "Model attempt succeeded | "
                "model=%s | "
                "attempt=%s",
                model_name,
                attempt_number,
            )


            return (
                response,
                attempt_number,
            )


        except Exception as error:

            retryable = (
                is_retryable_error(
                    error
                )
            )


            logger.warning(
                "Model attempt failed | "
                "model=%s | "
                "attempt=%s/%s | "
                "error=%s | "
                "retryable=%s",
                model_name,
                attempt_number,
                MAX_RETRIES + 1,
                type(error).__name__,
                retryable,
            )


            # ---------------------------------------------
            # Do NOT retry permanent errors
            # ---------------------------------------------

            if not retryable:

                raise ModelCallFailed(
                    model_name=model_name,
                    attempts=attempt_number,
                    original_error=error,
                ) from error


            # ---------------------------------------------
            # All retries exhausted
            # ---------------------------------------------

            if attempt_index == MAX_RETRIES:

                raise ModelCallFailed(
                    model_name=model_name,
                    attempts=attempt_number,
                    original_error=error,
                ) from error


            # ---------------------------------------------
            # Exponential Backoff
            # ---------------------------------------------

            retry_delay = (
                calculate_retry_delay(
                    attempt_index
                )
            )


            logger.warning(
                "Retry scheduled | "
                "model=%s | "
                "next_attempt=%s/%s | "
                "waiting=%.2fs",
                model_name,
                attempt_number + 1,
                MAX_RETRIES + 1,
                retry_delay,
            )


            await asyncio.sleep(
                retry_delay
            )


    raise RuntimeError(
        "Unexpected retry loop exit."
    )


# ---------------------------------------------------------
# Generate Answer
# ---------------------------------------------------------

async def generate_answer(
    message: str,
) -> GeneratedAnswer:

    overall_start_time = (
        time.perf_counter()
    )


    # -----------------------------------------------------
    # Try Primary Model
    # -----------------------------------------------------

    try:

        (
            response,
            primary_attempts,
        ) = await call_model_with_retries(

            model_name=PRIMARY_MODEL_NAME,

            message=message,
        )


        latency = (
            time.perf_counter()
            - overall_start_time
        )


        return GeneratedAnswer(

            message=message,

            response=response.output_text,

            model=PRIMARY_MODEL_NAME,

            fallback_used=False,

            input_tokens=(
                response.usage.input_tokens
            ),

            output_tokens=(
                response.usage.output_tokens
            ),

            total_tokens=(
                response.usage.total_tokens
            ),

            attempts=primary_attempts,

            latency_seconds=round(
                latency,
                2,
            ),
        )


    # -----------------------------------------------------
    # Primary Model Failed
    # -----------------------------------------------------

    except ModelCallFailed as primary_failure:

        logger.error(
            "Primary model failed | "
            "model=%s | "
            "attempts=%s | "
            "error=%s",
            primary_failure.model_name,
            primary_failure.attempts,
            type(
                primary_failure.original_error
            ).__name__,
        )


        # -------------------------------------------------
        # Only fallback for transient/retryable failures
        # -------------------------------------------------

        can_use_fallback = (

            FALLBACK_MODEL_NAME is not None

            and

            FALLBACK_MODEL_NAME
            != PRIMARY_MODEL_NAME

            and

            is_retryable_error(
                primary_failure.original_error
            )
        )


        if not can_use_fallback:

            raise


        logger.warning(
            "Switching to fallback model | "
            "primary=%s | "
            "fallback=%s",
            PRIMARY_MODEL_NAME,
            FALLBACK_MODEL_NAME,
        )


        # -------------------------------------------------
        # Try Fallback Model
        # -------------------------------------------------

        try:

            (
                fallback_response,
                fallback_attempts,
            ) = await call_model_with_retries(

                model_name=FALLBACK_MODEL_NAME,

                message=message,
            )


        except ModelCallFailed as fallback_failure:

            logger.error(
                "Fallback model also failed | "
                "model=%s | "
                "attempts=%s | "
                "error=%s",
                fallback_failure.model_name,
                fallback_failure.attempts,
                type(
                    fallback_failure.original_error
                ).__name__,
            )

            raise


        latency = (
            time.perf_counter()
            - overall_start_time
        )


        total_attempts = (
            primary_failure.attempts
            + fallback_attempts
        )


        return GeneratedAnswer(

            message=message,

            response=(
                fallback_response.output_text
            ),

            model=FALLBACK_MODEL_NAME,

            fallback_used=True,

            input_tokens=(
                fallback_response
                .usage
                .input_tokens
            ),

            output_tokens=(
                fallback_response
                .usage
                .output_tokens
            ),

            total_tokens=(
                fallback_response
                .usage
                .total_tokens
            ),

            attempts=total_attempts,

            latency_seconds=round(
                latency,
                2,
            ),
        )


# ---------------------------------------------------------
# Convert Failure Into Safe Batch Result
# ---------------------------------------------------------

def create_failed_batch_item(
    message: str,
    error: Exception,
) -> BatchItemResult:

    if isinstance(
        error,
        ModelCallFailed,
    ):

        actual_error = (
            error.original_error
        )

    else:

        actual_error = error


    return BatchItemResult(

        message=message,

        success=False,

        result=None,

        error_type=type(
            actual_error
        ).__name__,

        # Do not expose raw provider error details
        # to the API consumer.
        error_message=(
            "The AI request could not "
            "be completed."
        ),
    )


# ---------------------------------------------------------
# Convert Failure Into HTTP Error
# ---------------------------------------------------------

def raise_http_error(
    failure: ModelCallFailed,
):

    error = failure.original_error


    if isinstance(
        error,
        (
            TimeoutError,
            APITimeoutError,
        ),
    ):

        raise HTTPException(
            status_code=504,
            detail=(
                "The AI service timed out "
                "after multiple attempts."
            ),
        )


    if isinstance(
        error,
        RateLimitError,
    ):

        raise HTTPException(
            status_code=429,
            detail=(
                "The AI service is temporarily "
                "rate limited."
            ),
        )


    if isinstance(
        error,
        APIConnectionError,
    ):

        raise HTTPException(
            status_code=503,
            detail=(
                "Unable to connect to "
                "the AI provider."
            ),
        )


    if isinstance(
        error,
        APIStatusError,
    ):

        if error.status_code >= 500:

            raise HTTPException(
                status_code=503,
                detail=(
                    "The AI provider is "
                    "temporarily unavailable."
                ),
            )


    raise HTTPException(
        status_code=500,
        detail=(
            "Failed to generate AI response."
        ),
    )


# ---------------------------------------------------------
# Health Check
# ---------------------------------------------------------

@app.get("/health")
async def health():

    return {

        "status": "healthy",

        "service": (
            "Day 3 Production LLM API"
        ),

        "primary_model": (
            PRIMARY_MODEL_NAME
        ),

        "fallback_model": (
            FALLBACK_MODEL_NAME
        ),

        "timeout_seconds": (
            REQUEST_TIMEOUT_SECONDS
        ),

        "max_retries": (
            MAX_RETRIES
        ),
    }


# ---------------------------------------------------------
# Single Chat Endpoint
# ---------------------------------------------------------

@app.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(
    request: ChatRequest,
):

    try:

        result = await generate_answer(
            request.message
        )


        logger.info(
            "Chat completed | "
            "model=%s | "
            "fallback=%s | "
            "attempts=%s | "
            "tokens=%s | "
            "latency=%.2fs",
            result.model,
            result.fallback_used,
            result.attempts,
            result.total_tokens,
            result.latency_seconds,
        )


        return ChatResponse(

            response=result.response,

            model=result.model,

            fallback_used=(
                result.fallback_used
            ),

            input_tokens=(
                result.input_tokens
            ),

            output_tokens=(
                result.output_tokens
            ),

            total_tokens=(
                result.total_tokens
            ),

            attempts=(
                result.attempts
            ),

            latency_seconds=(
                result.latency_seconds
            ),
        )


    except ModelCallFailed as failure:

        raise_http_error(
            failure
        )


    except HTTPException:

        raise


    except Exception:

        logger.exception(
            "Unexpected chat failure"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unexpected AI service failure."
            ),
        )


# ---------------------------------------------------------
# Resilient Batch Endpoint
# ---------------------------------------------------------

@app.post(
    "/batch-chat",
    response_model=BatchChatResponse,
)
async def batch_chat(
    request: BatchChatRequest,
):

    batch_start_time = (
        time.perf_counter()
    )


    batch_results: list[
        BatchItemResult
    ] = []


    # -----------------------------------------------------
    # Sequential Mode
    # -----------------------------------------------------

    if request.mode == "sequential":

        for message in request.messages:

            try:

                answer = (
                    await generate_answer(
                        message
                    )
                )


                batch_results.append(

                    BatchItemResult(

                        message=message,

                        success=True,

                        result=answer,
                    )
                )


            except Exception as error:

                logger.warning(
                    "Batch item failed | "
                    "message_length=%s | "
                    "error=%s",
                    len(message),
                    type(error).__name__,
                )


                batch_results.append(

                    create_failed_batch_item(
                        message=message,
                        error=error,
                    )
                )


    # -----------------------------------------------------
    # Concurrent Mode
    # -----------------------------------------------------

    else:

        tasks = [

            generate_answer(message)

            for message
            in request.messages
        ]


        # IMPORTANT:
        #
        # return_exceptions=True
        #
        # means one failed task does not
        # destroy the entire batch.

        raw_results = await asyncio.gather(

            *tasks,

            return_exceptions=True,
        )


        for (
            message,
            raw_result,
        ) in zip(
            request.messages,
            raw_results,
        ):


            if isinstance(
                raw_result,
                Exception,
            ):

                logger.warning(
                    "Concurrent batch item failed | "
                    "message_length=%s | "
                    "error=%s",
                    len(message),
                    type(raw_result).__name__,
                )


                batch_results.append(

                    create_failed_batch_item(

                        message=message,

                        error=raw_result,
                    )
                )


            else:

                batch_results.append(

                    BatchItemResult(

                        message=message,

                        success=True,

                        result=raw_result,
                    )
                )


    # -----------------------------------------------------
    # Batch Metrics
    # -----------------------------------------------------

    total_latency = (

        time.perf_counter()
        - batch_start_time
    )


    successful_count = sum(

        1

        for item
        in batch_results

        if item.success
    )


    failed_count = (

        len(batch_results)
        - successful_count
    )


    total_tokens = sum(

        item.result.total_tokens

        for item
        in batch_results

        if (
            item.success
            and
            item.result is not None
        )
    )


    logger.info(
        "Batch completed | "
        "mode=%s | "
        "requests=%s | "
        "successful=%s | "
        "failed=%s | "
        "tokens=%s | "
        "latency=%.2fs",
        request.mode,
        len(request.messages),
        successful_count,
        failed_count,
        total_tokens,
        total_latency,
    )


    return BatchChatResponse(

        mode=request.mode,

        request_count=len(
            request.messages
        ),

        successful_count=(
            successful_count
        ),

        failed_count=(
            failed_count
        ),

        total_tokens=(
            total_tokens
        ),

        total_latency_seconds=round(
            total_latency,
            2,
        ),

        results=batch_results,
    )