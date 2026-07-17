"""
Shipping Agent -- Exposed as a REST API (LangGraph equivalent of ADK's A2A)

ADK equivalent: to_a2a(shipping_agent, port=8001) -- A2A protocol over HTTP
LangGraph equivalent: FastAPI wrapping a create_react_agent, same HTTP transport
                      concept but using plain JSON instead of A2A protocol.

Run: uvicorn shipping_agent:app --port 8001
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

# --- Tools ---

def get_shipping_status(order_id: str) -> dict:
    """Get current shipping status for an order."""
    data = {
        "ORD-1001": {"carrier": "FedEx", "tracking": "FX-789456123", "status": "in_transit", "location": "Memphis, TN"},
        "ORD-1002": {"carrier": "UPS", "tracking": "1Z999AA10123456784", "status": "delivered", "location": "Customer doorstep"},
        "ORD-1003": {"carrier": "USPS", "tracking": "9400111899223100001234", "status": "processing", "location": "Warehouse"},
        "ORD-1004": {"carrier": "DHL", "tracking": "DHL-5678901234", "status": "out_for_delivery", "location": "Local facility"},
        "ORD-1005": {"carrier": "FedEx", "tracking": "FX-321654987", "status": "in_transit", "location": "Chicago, IL"},
    }
    return data.get(order_id, {"error": f"No shipping info for {order_id}"})

def get_estimated_delivery(order_id: str) -> dict:
    """Get estimated delivery date for an order."""
    estimates = {
        "ORD-1001": {"date": "2026-02-14", "window": "9 AM - 5 PM", "note": "Signature required"},
        "ORD-1002": {"date": "2026-02-11", "window": "Delivered", "note": "Left at front door"},
        "ORD-1003": {"date": "2026-02-18", "window": "TBD", "note": "Still processing"},
        "ORD-1004": {"date": "2026-02-13", "window": "12 - 4 PM", "note": "Out for delivery today"},
        "ORD-1005": {"date": "2026-02-15", "window": "10 AM - 6 PM", "note": "Standard delivery"},
    }
    return estimates.get(order_id, {"error": f"No estimate for {order_id}"})

# --- LangGraph Agent ---
# ADK equivalent: Agent(name=..., tools=[...])

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(model=MODEL, temperature=0)

shipping_react_agent = create_react_agent(
    llm,
    tools=[get_shipping_status, get_estimated_delivery],
    prompt="You are a shipping specialist. Use get_shipping_status and get_estimated_delivery to help customers track their packages.",
)

# --- FastAPI Service ---
# ADK equivalent: app = to_a2a(shipping_agent, port=8001)
# LangGraph equivalent: FastAPI with /invoke endpoint (same HTTP transport, simpler protocol)

app = FastAPI(title="Shipping Agent", description="LangGraph shipping agent served as a REST API")

class InvokeRequest(BaseModel):
    message: str

class TraceStep(BaseModel):
    author: str = "shipping_agent"
    type: str  # tool_call | tool_response | text
    tool: str | None = None
    args: dict | None = None
    result: str | None = None
    text: str | None = None

class InvokeResponse(BaseModel):
    response: str
    trace: list[TraceStep] = []


@app.get("/.well-known/agent-card.json")
def agent_card():
    """Agent discovery endpoint -- mirrors ADK's A2A agent card."""
    return {
        "name": "shipping_agent",
        "description": "Handles shipping and delivery tracking.",
        "version": "1.0.0",
        "skills": ["get_shipping_status", "get_estimated_delivery"],
    }

def _message_text(content) -> str:
    """Normalize model content (str or list of parts) to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif hasattr(part, "text"):
                parts.append(getattr(part, "text") or "")
        return "\n".join(p for p in parts if p)
    return str(content)


def _build_trace(messages: list) -> list[TraceStep]:
    """Extract tool calls / results / final text for UI display."""
    steps: list[TraceStep] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            continue
        if isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    steps.append(TraceStep(
                        type="tool_call",
                        tool=tc["name"],
                        args=tc.get("args", {}),
                    ))
            elif msg.content:
                steps.append(TraceStep(type="text", text=_message_text(msg.content)))
        elif isinstance(msg, ToolMessage):
            steps.append(TraceStep(
                type="tool_response",
                tool=msg.name or "tool",
                result=str(msg.content)[:800],
            ))
    return steps


@app.post("/invoke", response_model=InvokeResponse)
async def invoke(req: InvokeRequest):
    """Run the shipping agent with the given message and return its response."""
    result = shipping_react_agent.invoke({"messages": [HumanMessage(content=req.message)]})
    messages = result["messages"]
    trace = _build_trace(messages)
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            return InvokeResponse(response=_message_text(msg.content), trace=trace)
    return InvokeResponse(response="(no response)", trace=trace)

# Run with: uvicorn shipping_agent:app --port 8001
