"""Pinecone vector store — config from env, OpenAI embeddings."""

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from load_env import make_openai_client

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100


@dataclass(frozen=True)
class PineconeSettings:
    api_key: str
    index_name: str
    namespace: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int


def get_pinecone_settings() -> PineconeSettings:
    api_key = os.getenv("PINECONE_API_KEY", "").strip()
    index_name = os.getenv("PINECONE_INDEX_NAME", "").strip()
    if not api_key:
        raise ValueError("PINECONE_API_KEY is not set")
    if not index_name:
        raise ValueError("PINECONE_INDEX_NAME is not set")

    return PineconeSettings(
        api_key=api_key,
        index_name=index_name,
        namespace=os.getenv("PINECONE_NAMESPACE", "").strip(),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", EMBEDDING_MODEL).strip(),
        chunk_size=int(os.getenv("INGEST_CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE))),
        chunk_overlap=int(os.getenv("INGEST_CHUNK_OVERLAP", str(DEFAULT_CHUNK_OVERLAP))),
    )


@lru_cache(maxsize=1)
def get_pinecone_index():
    from pinecone import Pinecone

    settings = get_pinecone_settings()
    pc = Pinecone(api_key=settings.api_key)
    return pc.Index(settings.index_name)


def embed_texts(texts: list[str], *, model: str | None = None) -> list[list[float]]:
    """Embed one or more strings with the configured OpenAI embedding model."""

    if not texts:
        return []

    settings = get_pinecone_settings()
    embedding_model = model or settings.embedding_model
    client = make_openai_client()
    response = client.embeddings.create(model=embedding_model, input=texts)
    return [item.embedding for item in response.data]


def upsert_texts(items: list[tuple[str, str, dict[str, Any] | None]]) -> int:
    """Embed and upsert (id, text, metadata) rows into Pinecone."""

    if not items:
        return 0

    settings = get_pinecone_settings()
    embeddings = embed_texts([text for _, text, _ in items])
    vectors = [
        (vec_id, values, metadata or {})
        for (vec_id, _text, metadata), values in zip(items, embeddings)
    ]

    index = get_pinecone_index()
    result = index.upsert(vectors=vectors, namespace=settings.namespace)
    return int(result.upserted_count)


def chunk_text(text: str) -> list[str]:
    """Split document text into overlapping chunks using env-configured sizes."""

    settings = get_pinecone_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_text(text)


def ingest_document(text: str, document_id: str, source: str | None = None) -> int:
    """Chunk, embed, and upsert a document into Pinecone."""

    chunks = chunk_text(text)
    if not chunks:
        return 0

    items: list[tuple[str, str, dict[str, Any] | None]] = []
    for chunk_index, chunk in enumerate(chunks):
        metadata: dict[str, Any] = {
            "document_id": document_id,
            "chunk_index": chunk_index,
            "source": source or "",
            "text": chunk,
        }
        vector_id = f"{document_id}::{chunk_index}"
        items.append((vector_id, chunk, metadata))

    return upsert_texts(items)


def query_text(
    text: str,
    top_k: int = 3,
    *,
    document_id: str | None = None,
) -> list[dict[str, Any]]:
    """Embed a query string and return the nearest neighbors from Pinecone."""

    settings = get_pinecone_settings()
    vector = embed_texts([text])[0]
    index = get_pinecone_index()

    query_kwargs: dict[str, Any] = {
        "vector": vector,
        "top_k": top_k,
        "namespace": settings.namespace,
        "include_metadata": True,
    }
    if document_id:
        query_kwargs["filter"] = {"document_id": {"$eq": document_id}}

    response = index.query(**query_kwargs)
    return [
        {"id": match.id, "score": match.score, "metadata": match.metadata or {}}
        for match in response.matches
    ]


def check_pinecone_health() -> dict[str, Any]:
    """Confirm Pinecone config is present and the index responds."""

    try:
        settings = get_pinecone_settings()
    except ValueError as exc:
        return {"status": "error", "reachable": False, "detail": str(exc)}

    try:
        index = get_pinecone_index()
        stats = index.describe_index_stats()
        dimension_ok = stats.dimension == EMBEDDING_DIMENSION
        namespaces = {
            name: summary.vector_count for name, summary in (stats.namespaces or {}).items()
        }

        return {
            "status": "ok" if dimension_ok else "warning",
            "reachable": True,
            "index_name": settings.index_name,
            "namespace": settings.namespace or "(default)",
            "embedding_model": settings.embedding_model,
            "expected_dimension": EMBEDDING_DIMENSION,
            "index_dimension": stats.dimension,
            "dimension_matches_embedding_model": dimension_ok,
            "total_vector_count": stats.total_vector_count,
            "namespaces": namespaces,
        }
    except Exception as exc:
        return {
            "status": "error",
            "reachable": False,
            "index_name": settings.index_name,
            "embedding_model": settings.embedding_model,
            "detail": str(exc),
        }
