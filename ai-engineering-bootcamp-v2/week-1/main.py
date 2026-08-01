"""Week 1 live demo — five stages in one file, built up live in class."""

import os
import time
from typing import Any

from load_env import load_course_env, make_openai_client
from fastapi import FastAPI, HTTPException, Query
from pinecone_store import check_pinecone_health, ingest_document, query_text
from pydantic import BaseModel, Field, ValidationError

# Load .env from this folder, then fall back to the course root if needed.
load_course_env()

# Reuse one client so TLS handshakes are not repeated on every request.
app = FastAPI()
client = make_openai_client()

# Stage 4 default — strong general model; swap at request time for the live demo.
DEFAULT_MODEL = "gpt-4o"

# Stage 5 — per-1K-token input/output USD (derived from OpenAI list prices).
MODEL_PRICES_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "o3-mini": (0.0011, 0.0044),
}

DEFAULT_RETRIEVE_TOP_K = int(os.getenv("RETRIEVE_TOP_K", "5"))

GROUNDING_PROMPT_TEMPLATE = """Answer using ONLY the context below.
If the context does not contain the answer, say:
"I don't have enough information to answer that."
Cite the document_id of each chunk you used.

Context:
{retrieved_chunks}

Question: {question}"""


class Answer(BaseModel):
    """Structured model output — this is what turns a chatbot into a component."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources_needed: bool


class AskRequest(BaseModel):
    """Typed request body so bad input is rejected before we spend tokens."""

    question: str
    force_bad: bool = False  # Stage 3 demo knob — first attempt breaks schema on purpose.
    model: str | None = None  # Stage 4 — optional override to swap models live.


class AskResponse(BaseModel):
    """Typed response so callers always get the same shape back."""

    answer: Answer
    tokens_used: int
    model: str
    latency_ms: int
    cost_usd: float
    retrieved_chunk_ids: list[str] = Field(default_factory=list)


class IngestRequest(BaseModel):
    """Document payload for indexing into Pinecone."""

    text: str
    document_id: str
    source: str | None = None  # Optional filename or URL label.


class IngestResponse(BaseModel):
    document_id: str
    chunks_indexed: int
    status: str


class RetrieveChunk(BaseModel):
    id: str
    score: float
    document_id: str | None = None
    chunk_index: int | None = None
    source: str | None = None


class RetrieveResponse(BaseModel):
    query: str
    top_k: int
    results: list[RetrieveChunk]


def format_retrieved_chunks(matches: list[dict[str, Any]]) -> str:
    """Turn Pinecone matches into labeled context blocks for the grounding prompt."""

    blocks: list[str] = []
    for match in matches:
        metadata = match.get("metadata") or {}
        chunk_text = (metadata.get("text") or "").strip()
        if not chunk_text:
            continue

        document_id = metadata.get("document_id", "unknown")
        blocks.append(
            "\n".join(
                [
                    f"[document_id: {document_id} | chunk_id: {match['id']}]",
                    chunk_text,
                ]
            )
        )

    return "\n\n".join(blocks)


def build_grounding_prompt(question: str, matches: list[dict[str, Any]]) -> str:
    """Build the RAG prompt that constrains the model to retrieved context."""

    retrieved_chunks = format_retrieved_chunks(matches)
    if not retrieved_chunks:
        retrieved_chunks = "(no chunk text available — re-ingest documents to store chunk text in metadata)"

    return GROUNDING_PROMPT_TEMPLATE.format(
        retrieved_chunks=retrieved_chunks,
        question=question,
    )


def compute_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Turn real usage into dollars — same prompt, different model, different cost."""

    prices = MODEL_PRICES_PER_1K.get(model, MODEL_PRICES_PER_1K[DEFAULT_MODEL])
    input_per_1k, output_per_1k = prices
    return (prompt_tokens / 1000 * input_per_1k) + (completion_tokens / 1000 * output_per_1k)


def call_model_structured(question: str, model: str) -> tuple[Answer, int, int, int]:
    """
    Stage 2 center: OpenAI structured output forces exactly the Answer schema.
    Returns parsed answer plus token counts from billing metadata.
    """

    completion = client.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": question}],
        response_format=Answer,
    )

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Model returned no parseable structured output")

    usage = completion.usage
    total = usage.total_tokens if usage else 0
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    return parsed, total, prompt_tokens, completion_tokens


def call_model_unsafe(question: str, model: str) -> tuple[Answer, int, int, int]:
    """
    Stage 3 demo path: free-form JSON call, then validate locally.
    The bad instruction makes confidence a string so Pydantic rejects it reliably.
    """

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{question}\n\n"
                    "Reply with ONLY a JSON object using keys answer, confidence, sources_needed. "
                    "Set confidence to the string 'very high' (not a number)."
                ),
            }
        ],
    )

    raw = completion.choices[0].message.content or ""
    # Guardrail: refuse malformed output instead of passing it through to clients.
    answer = Answer.model_validate_json(raw)

    usage = completion.usage
    total = usage.total_tokens if usage else 0
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    return answer, total, prompt_tokens, completion_tokens


@app.post("/ask")
def ask(body: AskRequest) -> AskResponse:
    """Answer one question with RAG, structured output, guardrails, and cost visibility."""

    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    model = body.model or DEFAULT_MODEL
    last_error: str | None = None

    try:
        matches = query_text(question, top_k=DEFAULT_RETRIEVE_TOP_K)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Retrieval failed: {exc}") from exc

    retrieved_chunk_ids = [match["id"] for match in matches]
    grounding_prompt = build_grounding_prompt(question, matches)

    if matches and not format_retrieved_chunks(matches):
        raise HTTPException(
            status_code=503,
            detail=(
                "Retrieved chunks have no stored text. Re-ingest documents so chunk text "
                "is saved in Pinecone metadata."
            ),
        )

    # Stage 3: one retry keeps the logic legible while still protecting callers.
    for attempt in range(2):
        try:
            start = time.perf_counter()

            # First attempt with force_bad uses the unsafe path; retry uses structured output.
            use_bad_path = body.force_bad and attempt == 0
            if use_bad_path:
                answer, tokens_used, prompt_tokens, completion_tokens = call_model_unsafe(
                    grounding_prompt, model
                )
            else:
                answer, tokens_used, prompt_tokens, completion_tokens = call_model_structured(
                    grounding_prompt, model
                )

            latency_ms = int((time.perf_counter() - start) * 1000)
            cost_usd = compute_cost_usd(model, prompt_tokens, completion_tokens)

            return AskResponse(
                answer=answer,
                tokens_used=tokens_used,
                model=model,
                latency_ms=latency_ms,
                cost_usd=round(cost_usd, 6),
                retrieved_chunk_ids=retrieved_chunk_ids,
            )
        except (ValidationError, ValueError) as exc:
            last_error = str(exc)
            continue

    # Clean failure — never leak a half-parsed response to the client.
    raise HTTPException(
        status_code=502,
        detail=f"Model response failed schema validation after retry: {last_error}",
    )


@app.get("/health/pinecone")
def pinecone_health():
    """Debug endpoint — confirms Pinecone env vars and index connectivity."""

    return check_pinecone_health()


@app.get("/debug/retrieve")
def debug_retrieve(
    q: str = Query(..., min_length=1, description="Question or search phrase to embed"),
    top_k: int = Query(5, ge=1, le=20, description="Number of nearest chunks to return"),
) -> RetrieveResponse:
    """Embed a query and return Pinecone matches — no LLM call.

    Example:
        curl -s "http://127.0.0.1:8000/debug/retrieve?q=What%20is%20Newton%27s%20second%20law"
    """

    query = q.strip()
    if not query:
        raise HTTPException(status_code=400, detail="q must not be empty")

    try:
        matches = query_text(query, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Retrieval failed: {exc}") from exc

    results = [
        RetrieveChunk(
            id=match["id"],
            score=match["score"],
            document_id=match["metadata"].get("document_id"),
            chunk_index=match["metadata"].get("chunk_index"),
            source=match["metadata"].get("source") or None,
        )
        for match in matches
    ]

    return RetrieveResponse(query=query, top_k=top_k, results=results)


@app.post("/ingest")
def ingest(body: IngestRequest) -> IngestResponse:
    """Chunk, embed, and store a document in Pinecone.

    Example:
        curl -s -X POST http://127.0.0.1:8000/ingest \\
          -H "Content-Type: application/json" \\
          -d '{
            "document_id": "rag-intro",
            "source": "notes.txt",
            "text": "Retrieval-Augmented Generation combines search with language models."
          }'
    """

    text = body.text.strip()
    document_id = body.document_id.strip()

    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")
    if not document_id:
        raise HTTPException(status_code=400, detail="document_id must not be empty")

    try:
        chunks_indexed = ingest_document(text, document_id, source=body.source)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ingest failed: {exc}") from exc

    if chunks_indexed == 0:
        raise HTTPException(status_code=400, detail="no chunks produced from text")

    return IngestResponse(
        document_id=document_id,
        chunks_indexed=chunks_indexed,
        status="indexed",
    )
