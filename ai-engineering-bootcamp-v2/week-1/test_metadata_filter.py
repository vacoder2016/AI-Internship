#!/usr/bin/env python3
"""Before/after test for metadata filtering on retrieval.

Usage:
  python test_metadata_filter.py
  python test_metadata_filter.py --base-url https://ai-internship-cub8.onrender.com
"""

import argparse
import json
import ssl
import sys

import httpx
import truststore

TEST_QUERY = "What is the unit of force in SI"
FILTER_DOC = "newtons-laws"


def fetch(base_url: str, document_id: str | None) -> dict:
    params: dict = {"q": TEST_QUERY, "top_k": 5}
    if document_id:
        params["document_id"] = document_id

    kwargs: dict = {"timeout": 60.0}
    if base_url.startswith("https://"):
        kwargs["verify"] = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

    with httpx.Client(**kwargs) as client:
        response = client.get(f"{base_url.rstrip('/')}/debug/retrieve", params=params)
        response.raise_for_status()
        return response.json()


def summarize(label: str, data: dict) -> None:
    ids = [r["id"] for r in data.get("results", [])]
    doc_ids = [r.get("document_id") for r in data.get("results", [])]
    foreign = [cid for cid, did in zip(ids, doc_ids) if did and did != FILTER_DOC]

    print(f"\n{label}")
    print(f"  document_id_filter: {data.get('document_id_filter')}")
    print(f"  top-5 chunk IDs:    {ids}")
    print(f"  document_ids:       {doc_ids}")
    if foreign:
        print(f"  foreign chunks:     {foreign}  <-- noise from other docs")
    else:
        print("  foreign chunks:     none")


def main() -> int:
    parser = argparse.ArgumentParser(description="Test metadata filtering before/after")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    try:
        before = fetch(args.base_url, document_id=None)
        after = fetch(args.base_url, document_id=FILTER_DOC)
    except httpx.HTTPError as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1

    print(f"Query: {TEST_QUERY}")
    print(f"API:   {args.base_url}")

    summarize("BEFORE (no filter)", before)
    summarize(f"AFTER  (document_id={FILTER_DOC})", after)

    before_foreign = any(
        r.get("document_id") not in (None, FILTER_DOC) for r in before.get("results", [])
    )
    after_foreign = any(
        r.get("document_id") not in (None, FILTER_DOC) for r in after.get("results", [])
    )

    print("\n" + "=" * 50)
    if before_foreign and not after_foreign:
        print("PASS — filter removed cross-document noise from top-5.")
        return 0
    if not before_foreign:
        print("NOTE — no foreign chunks before filter; index may only have one doc.")
        return 0
    print("CHECK — foreign chunks still present after filter.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
