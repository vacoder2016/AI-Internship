# Week 1 — `/ask` Demo + Session 2 RAG

Build a typed LLM endpoint step by step, then add retrieval-augmented generation on top.

## Setup

```bash
cp .env.example .env          # OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME
python -m venv .venv
source .venv/bin/activate     # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Demo stages (Session 1)

| Stage | File | What you learn |
|-------|------|----------------|
| 1 | `serve_stage1.py` | Bare `/ask` — string answer + `tokens_used` |
| 2 | `serve_stage2.py` | Structured output via Pydantic + `completions.parse` |
| 3 | `serve_stage3.py` | Validation guardrail + retry (`force_bad` demo knob) |
| 4 | `serve_stage4.py` | Per-request `model` override + `latency_ms` |
| 5 | `serve_stage5.py` / `main.py` | Full system + `cost_usd` readout |

Run one stage at a time (only one server on port 8000):

```bash
uvicorn serve_stage1:app --host 127.0.0.1 --port 8000 --reload
# or the full system:
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### Streamlit stage demo (Session 1)

Interactive UI for all five stages:

```bash
streamlit run demo_page.py
```

Open http://localhost:8501. Set **API base URL** to `http://127.0.0.1:8000` and start the matching stage server in another terminal.

---

## Session 2 — RAG API

`main.py` adds Pinecone-backed ingest and grounded `/ask`. The API is the source of truth — clients do not reimplement RAG.

| Endpoint | Purpose |
|----------|---------|
| `POST /ingest` | Chunk, embed, and store a document |
| `GET /debug/retrieve?q=...` | Embed a query and return top-k chunks (no LLM) |
| `POST /ask` | Retrieve context, build grounding prompt, return structured answer |
| `GET /health/pinecone` | Confirm Pinecone config and index connectivity |

### Ingest

```bash
curl -s -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "newtons-laws",
    "source": "NewtonsLaws.pdf",
    "text": "Your document text here..."
  }'
```

Or ingest a PDF locally:

```bash
python ingest_pdfs.py NewtonsLaws.pdf
```

### Debug retrieval (no LLM)

```bash
curl -s "http://127.0.0.1:8000/debug/retrieve?q=What%20is%20Newton%27s%20second%20law&top_k=5"
```

### Ask (RAG)

```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Newton'\''s second law?"}'
```

Response includes `retrieved_chunk_ids` and structured `answer` (`answer`, `confidence`, `sources_needed`).

### Grounding prompt

`/ask` constrains the model with:

```text
Answer using ONLY the context below.
If the context does not contain the answer, say:
"I don't have enough information to answer that."
Cite the document_id of each chunk you used.

Context:
{retrieved_chunks}

Question: {question}
```

---

## Streamlit RAG UI (Session 2)

Minimal client for `POST /ingest` and `POST /ask`:

```bash
streamlit run rag_ui.py
```

Set **API base URL** in the sidebar (local default: `http://127.0.0.1:8000`). Optional env var:

```bash
RAG_API_BASE_URL=http://127.0.0.1:8000
```

Tabs:

- **Ingest** — paste text + `document_id`, call `/ingest`
- **Ask** — question + optional `document_id` filter, call `/ask`, show citations or refusal

---

## Add-on: metadata filtering

When multiple documents share one Pinecone index, unfiltered search can return chunks from the wrong doc. Optional `document_id` metadata filtering scopes retrieval to one document.

**API**

```bash
# Unfiltered (may include chunks from other docs)
curl -s "http://127.0.0.1:8000/debug/retrieve?q=What%20is%20the%20unit%20of%20force%20in%20SI&top_k=5"

# Filtered to newtons-laws only
curl -s "http://127.0.0.1:8000/debug/retrieve?q=What%20is%20the%20unit%20of%20force%20in%20SI&top_k=5&document_id=newtons-laws"
```

```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the unit of force in SI?", "document_id": "newtons-laws"}'
```

**Before/after test**

```bash
python test_metadata_filter.py
# optional: python test_metadata_filter.py --base-url http://127.0.0.1:8000
```

Example result: before filter, top-5 may include `ragintro::0`; after `document_id=newtons-laws`, all hits are `newtons-laws::…`.

---

## Golden-set eval

**Corpus:** `NewtonsLaws.pdf` → `document_id: newtons-laws`

**Eval method:** `GET /debug/retrieve?q=...&top_k=5` for retrieval; `POST /ask` for answers.

| Question | Expected answer (short) | Retrieval hit? (right chunk in top-5) | Faithfulness? (grounded in retrieved text) | Correctness? (matches expected) |
|----------|-------------------------|---------------------------------------|--------------------------------------------|----------------------------------|
| What is Newton's second law? | F = ma — force equals mass × acceleration | **Yes** (`newtons-laws::5` in top-5) | **Yes** | **Yes** |
| What is Newton's first law? | Object stays at rest or uniform motion unless a net force acts | **Yes** (`newtons-laws::1`, `::2`, `::3` in top-5) | **Yes** | **Yes** |
| What is inertia? | Resistance to changes in motion | **Yes** (`newtons-laws::7`, `::2` in top-5) | **Yes** | **Yes** |
| What is the unit of force in SI? | The newton (N) | **Yes** (`newtons-laws::6`, `::7` in top-5) | **Yes** | **Yes** |
| What is Newton's third law? | Equal and opposite action–reaction forces | **No** (ideal chunks `::8`–`::10` missed; top-5 were `::5`, `::12`, `::1`…) | **Yes** (still grounded in retrieved Newton text) | **Yes** |
| How do you bake a chocolate cake? | **Refusal** — not in docs | **No** (no relevant chunk; top hits are unrelated `newtons-laws` chunks) | **Yes** (returns refusal phrase) | **Yes** (exact refusal: *"I don't have enough information to answer that."*) |

**Score (6 cases):** Retrieval 4/6 · Faithfulness 6/6 · Correctness 6/6

**Known gap:** Third-law retrieval ranks general Newton chunks ahead of the dedicated third-law section — answer is still correct, but chunking/retrieval could be improved.

Questions are stored in `golden_set.json` for re-runs.

### Column definitions

| Column | Pass criteria |
|--------|----------------|
| **Retrieval hit?** | An ideal chunk for that topic appears in top-5 (`newtons-laws::N`) |
| **Faithfulness?** | Answer uses retrieved context only; refusal uses the configured phrase |
| **Correctness?** | Answer matches the short expected fact (or refusal for off-doc question) |

---

## Smoke-test Session 1 stages

Requires `.venv` and a valid `OPENAI_API_KEY`:

```bash
python test_all_stages.py
```

---

## Project layout

```
week-1/
├── main.py                  # Full system: Session 1 + Session 2 RAG
├── pinecone_store.py        # Pinecone embed, ingest, query, health
├── serve_stage1.py … serve_stage5.py
├── demo_page.py             # Streamlit UI for Session 1 stages
├── rag_ui.py                # Streamlit UI for Session 2 ingest + ask
├── ingest_pdfs.py           # PDF → POST /ingest helper
├── test_metadata_filter.py  # Before/after metadata filter test
├── golden_set.json          # Golden-set questions
├── test_all_stages.py       # Automated Session 1 stage smoke tests
├── requirements.txt
├── .env.example
└── .gitignore
```
