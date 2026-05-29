# Week 1 — FastAPI + OpenAI

## Run locally

```bash
cp .env.example .env        # add your OPENAI_API_KEY
pip install -r requirements.txt
fastapi dev main.py
```

## Test

```bash
# Health check
curl http://localhost:8000/

# Ask (structured response)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is FastAPI?"}'

# Ask with context
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarise this", "context": "FastAPI is a modern Python web framework."}'

# Stream
curl -X POST http://localhost:8000/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain Python in 3 sentences."}'
```
