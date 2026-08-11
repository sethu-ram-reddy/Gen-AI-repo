import hashlib
import logging
import os 
import time
import uuid

from dataclasses import dataclass, field
from dotenv import load_dotenv
from openai import AsyncOpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Configuration
load_dotenv()

MODEL_NAME = os.getenv(
    "OPENAI_MODEL",
    "gpt-5.6-luna",
)
MAX_ACTIVE_TURNS = 6

CACHE_TTL_SECONDS = 300

client = AsyncOpenAI()

app = FastAPI(
    title = "Day 3 - State, Context and Cache",
    description = (
        "Learning project for conversation state, "
        "context management, caching and observability."
    ),
    version = "1.0.0",
)

# Logging
logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Conversation State

@dataclass
class ConversationState:

    # ID of the latest openAI response
    previous_response_id: str | None = None

    # Human redable transcript stored y our application
    transcript: list[dict[str, str]] = Field(
        default_factory = list
    )
    # Summary of the configuration after completition
    summary: str | None = None

    # Number of turns since last compaction
    active_turns: int = 0

    # Total turns during the entire conversation
    total_turns: int = 0

# For the learning purpose only: For the prod we usee the technologie like Redis, PostGREsql or another persistent storage
conversation_store: dict[str, ConversationState] = {}

# Cache State
@dataclass
class CacheEntry:
    response: str
    created_at: float

response_cache: dict[str, CacheEntry] = {}

# API Models
class ConversationCreateResponse(BaseModel):
    conversation_id:str

class ConversationChatRequest(BaseModel):
    conversation_id:str
    message: str = Field(
        min_length = 1,
        max_length = 4000,
    )

class ConversationChatResponse(BaseModel):
    conversation_id: str
    response: str
    active_turns: int
    total_turns: int
    context_compacted: bool
    latency_seconds: float

class CachedChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=4000,
    )

class CachedChatResponse(BaseModel):
    response: str
    cache_hit: bool
    latency_seconds: float


BASE_INSTRUCTIONS = """
You are an helpful AI engineering assistant.

Rules:
- Be technically accurate.
- Answer clearly and accurately.
- Remember the relevant information from the conversation
- Do not pretend to remember the information that was never provided.
"""

# API Health Check
@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "active_conversations": len(
            conversation_store
        ),
        "cached_items": len(
            response_cache
        ),
    }

# Create Conversation
@app.post(
    "/conversations",
    response_model=ConversationCreateResponse,
)
async def create_conversation():

    conversation_id = str(
        uuid.uuid4()
    )

    conversation_store[
        conversation_id
    ] = ConversationState()

    logger.info(
        "Conversation created | "
        "conversation_id=%s",
        conversation_id,
    )

    return ConversationCreateResponse(
        conversation_id=conversation_id
    )

# Context Compaction
async def compact_conversation(
    state: ConversationState,
) -> None:

    if not state.transcript:
        return


    transcript_text = "\n".join(

        f"{item['role']}: {item['content']}"

        for item in state.transcript
    )


    response = await client.responses.create(

        model=MODEL_NAME,

        instructions="""
        Summarize the following conversation for future
        conversational context.

        Preserve only useful information such as:
        - important facts
        - user preferences
        - decisions
        - unresolved questions
        - project context

        Remove:
        - repetition
        - greetings
        - unnecessary wording

        Keep the summary concise.
        """,

        input=transcript_text,
    )


    state.summary = (
        response.output_text
    )


    # We are starting a new response chain.
    # Old context is now represented by the summary.
    state.previous_response_id = None


    # Clear old raw transcript.
    state.transcript.clear()


    # Reset active context counter.
    state.active_turns = 0


    logger.info(
        "Conversation context compacted | "
        "summary_length=%s",
        len(state.summary),
    )

# Conversation Chat
@app.post(
    "/conversation-chat",
    response_model=ConversationChatResponse,
)
async def conversation_chat(
    request: ConversationChatRequest,
):

    request_id = str(
        uuid.uuid4()
    )

    start_time = (
        time.perf_counter()
    )


    state = conversation_store.get(
        request.conversation_id
    )


    if state is None:

        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )


    context_compacted = False

    # Context Management
    if (
        state.active_turns
        >= MAX_ACTIVE_TURNS
    ):

        await compact_conversation(
            state
        )

        context_compacted = True


    # Build Instructions
    instructions = (
        BASE_INSTRUCTIONS
    )


    if state.summary:

        instructions += f"""

Conversation summary from earlier turns:

{state.summary}
"""


    # Call Model
    try:

        if state.previous_response_id:

            response = await client.responses.create(

                model=MODEL_NAME,

                instructions=instructions,

                previous_response_id=(
                    state.previous_response_id
                ),

                input=request.message,
            )

        else:

            response = await client.responses.create(

                model=MODEL_NAME,

                instructions=instructions,

                input=request.message,
            )


    except Exception:

        logger.exception(
            "Conversation request failed | "
            "request_id=%s",
            request_id,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to generate conversation response."
            ),
        )

    # Save State
    state.previous_response_id = (
        response.id
    )


    state.transcript.append(
        {
            "role": "user",
            "content": request.message,
        }
    )


    state.transcript.append(
        {
            "role": "assistant",
            "content": response.output_text,
        }
    )


    state.active_turns += 1

    state.total_turns += 1


    latency = (
        time.perf_counter()
        - start_time
    )


    logger.info(
        "Conversation request completed | "
        "request_id=%s | "
        "conversation_id=%s | "
        "active_turns=%s | "
        "total_turns=%s | "
        "compacted=%s | "
        "latency=%.2fs",
        request_id,
        request.conversation_id,
        state.active_turns,
        state.total_turns,
        context_compacted,
        latency,
    )


    return ConversationChatResponse(

        conversation_id=(
            request.conversation_id
        ),

        response=(
            response.output_text
        ),

        active_turns=(
            state.active_turns
        ),

        total_turns=(
            state.total_turns
        ),

        context_compacted=(
            context_compacted
        ),

        latency_seconds=round(
            latency,
            2,
        ),
    )


# Inspect Conversation
@app.get(
    "/conversations/{conversation_id}"
)
async def get_conversation(
    conversation_id: str,
):

    state = conversation_store.get(
        conversation_id
    )


    if state is None:

        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )


    return {

        "conversation_id": (
            conversation_id
        ),

        "previous_response_id": (
            state.previous_response_id
        ),

        "summary": (
            state.summary
        ),

        "active_turns": (
            state.active_turns
        ),

        "total_turns": (
            state.total_turns
        ),

        "transcript": (
            state.transcript
        ),
    }


# Delete Conversation
@app.delete(
    "/conversations/{conversation_id}"
)
async def delete_conversation(
    conversation_id: str,
):

    if (
        conversation_id
        not in conversation_store
    ):

        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )


    del conversation_store[
        conversation_id
    ]


    logger.info(
        "Conversation deleted | "
        "conversation_id=%s",
        conversation_id,
    )


    return {
        "deleted": True,
        "conversation_id": conversation_id,
    }


# Cache Helpers
def normalize_message(
    message: str,
) -> str:

    return " ".join(
        message
        .lower()
        .strip()
        .split()
    )


def create_cache_key(
    message: str,
) -> str:

    normalized = (
        normalize_message(
            message
        )
    )


    return hashlib.sha256(
        normalized.encode(
            "utf-8"
        )
    ).hexdigest()


def get_cached_response(
    cache_key: str,
) -> str | None:

    entry = response_cache.get(
        cache_key
    )


    if entry is None:
        return None


    age = (
        time.time()
        - entry.created_at
    )


    if age > CACHE_TTL_SECONDS:

        del response_cache[
            cache_key
        ]

        return None


    return entry.response

# Stateless Cached Chat
@app.post(
    "/cached-chat",
    response_model=CachedChatResponse,
)
async def cached_chat(
    request: CachedChatRequest,
):

    start_time = (
        time.perf_counter()
    )


    cache_key = (
        create_cache_key(
            request.message
        )
    )


    # CACHE HIT
    cached_response = (
        get_cached_response(
            cache_key
        )
    )


    if cached_response is not None:

        latency = (
            time.perf_counter()
            - start_time
        )


        logger.info(
            "Cache hit | "
            "key=%s | "
            "latency=%.4fs",
            cache_key[:12],
            latency,
        )


        return CachedChatResponse(

            response=cached_response,

            cache_hit=True,

            latency_seconds=round(
                latency,
                4,
            ),
        )


    # CACHE MISS

    logger.info(
        "Cache miss | key=%s",
        cache_key[:12],
    )


    try:

        response = await client.responses.create(

            model=MODEL_NAME,

            instructions=BASE_INSTRUCTIONS,

            input=request.message,
        )


    except Exception:

        logger.exception(
            "Cached chat LLM request failed"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to generate AI response."
            ),
        )


    # Store Response
    response_cache[
        cache_key
    ] = CacheEntry(

        response=response.output_text,

        created_at=time.time(),
    )


    latency = (
        time.perf_counter()
        - start_time
    )


    logger.info(
        "Cache populated | "
        "key=%s | "
        "tokens=%s | "
        "latency=%.2fs",
        cache_key[:12],
        response.usage.total_tokens,
        latency,
    )


    return CachedChatResponse(

        response=(
            response.output_text
        ),

        cache_hit=False,

        latency_seconds=round(
            latency,
            2,
        ),
    )