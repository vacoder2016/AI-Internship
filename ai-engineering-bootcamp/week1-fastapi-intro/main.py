from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
client = OpenAI(timeout=20.0, max_retries=3)


class Question(BaseModel):
    question: str
    context: str | None = None


class Answer(BaseModel):
    answer: str
    sources: list[str]
    confidence: float


@app.get("/")
def health():
    return {"status": "ok"}

@app.get("/health")
def health_check():
    return {"status": "health is okay"}

@app.post("/ask")
def ask(q: Question):
    messages = [{"role": "user", "content": q.question}]
    if q.context:
        messages.insert(0, {"role": "system", "content": q.context})

    try:
        completion = client.chat.completions.parse(
            model="gpt-5.4-mini",
            messages=messages,
            response_format=Answer,
        )
        return completion.choices[0].message.parsed
    except Exception:
        return Answer(answer="Something went wrong.", sources=[], confidence=0.0)


@app.post("/ask/stream")
def ask_stream(q: Question):
    messages = [{"role": "user", "content": q.question}]
    if q.context:
        messages.insert(0, {"role": "system", "content": q.context})

    def generate():
        stream = client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    return StreamingResponse(generate(), media_type="text/plain")
