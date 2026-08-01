#!/usr/bin/env python3
"""Ingest PDF files via POST /ingest and report chunk counts."""

import argparse
import re
import sys
from pathlib import Path

import httpx
from pypdf import PdfReader

DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def pdf_to_document_id(path: Path) -> str:
    """Stable id from filename: NewtonsLaws.pdf -> newtons-laws."""

    stem = path.stem
    normalized = re.sub(r"([a-z])([A-Z])", r"\1-\2", stem)
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", normalized)
    return normalized.lower().strip("-")


def ingest_file(client: httpx.Client, base_url: str, path: Path) -> int:
    document_id = pdf_to_document_id(path)
    reader = PdfReader(str(path))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    page_count = len(reader.pages)
    char_count = len(text)

    print(f"\n{path.name}")
    print(f"  document_id: {document_id}")
    print(f"  pages: {page_count}, characters: {char_count}")

    response = client.post(
        f"{base_url.rstrip('/')}/ingest",
        json={
            "document_id": document_id,
            "source": path.name,
            "text": text,
        },
        timeout=600.0,
    )

    if response.status_code != 200:
        print(f"  ERROR HTTP {response.status_code}: {response.text}")
        response.raise_for_status()

    data = response.json()
    chunks = int(data["chunks_indexed"])
    print(f"  chunks_indexed: {chunks}")
    return chunks


def print_vector_store_total(client: httpx.Client, base_url: str) -> None:
    response = client.get(f"{base_url.rstrip('/')}/health/pinecone", timeout=30.0)
    response.raise_for_status()
    stats = response.json()

    total = stats.get("total_vector_count")
    namespace = stats.get("namespace", "(default)")
    index_name = stats.get("index_name", "unknown")

    print("\n" + "=" * 40)
    print(f"Vector store: {index_name} (namespace: {namespace})")
    print(f"Total chunks in index: {total}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest PDFs via POST /ingest")
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="PDF paths (default: NewtonsLaws.pdf in this folder)",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"API base URL (default: {DEFAULT_BASE_URL})",
    )
    args = parser.parse_args()

    workdir = Path(__file__).resolve().parent
    files = args.files or [workdir / "NewtonsLaws.pdf"]

    missing = [path for path in files if not path.is_file()]
    if missing:
        for path in missing:
            print(f"File not found: {path}", file=sys.stderr)
        return 1

    total_ingested = 0
    with httpx.Client() as client:
        for path in files:
            total_ingested += ingest_file(client, args.base_url, path.resolve())

        print(f"\nIngested this run: {total_ingested} chunks")
        print_vector_store_total(client, args.base_url)

    return 0


if __name__ == "__main__":
    sys.exit(main())
