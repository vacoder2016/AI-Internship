"""
Demo 3: Full System -- Routing + MCP + Remote Agent

ADK equivalent: root_agent with sub_agents=[billing_agent(MCP), technical_agent, RemoteA2aAgent]
LangGraph equivalent: StateGraph supervisor routing to:
  - billing_agent  : create_react_agent + MCP tools (Supabase) via MultiServerMCPClient
  - technical_agent: create_react_agent + local tools
  - shipping_node  : HTTP call to shipping_agent FastAPI service (replaces A2A)

Start shipping agent first:  uvicorn shipping_agent:app --port 8001
Then run:                     python demo3_full_system.py
"""

import asyncio
import os
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

from langfuse.langchain import CallbackHandler as LangfuseHandler
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
import httpx

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TOKEN = os.getenv("SUPABASE_ACCESS_TOKEN", "")
PROJECT_REF = os.getenv("SUPABASE_PROJECT_REF", "")

if not TOKEN:
    print("WARNING: SUPABASE_ACCESS_TOKEN not set -- billing agent won't work.")

llm = ChatOpenAI(model=MODEL, temperature=0)

# --- Layer 1: Technical Agent (local tools) ---
# ADK equivalent: Agent(name="technical_agent", tools=[...])

def search_knowledge_base(query: str) -> dict:
    """Search the knowledge base for technical solutions."""
    articles = {
        "login": {"title": "Login Issues", "solution": "1. Clear cache. 2. Try incognito. 3. Reset password."},
        "crash": {"title": "App Crashing", "solution": "1. Update to v3.2.1. 2. Clear app data. 3. Check OS requirements."},
        "slow": {"title": "Performance Issues", "solution": "1. Check internet. 2. Close other apps. 3. Enable hardware acceleration."},
    }
    for keyword, article in articles.items():
        if keyword in query.lower():
            return article
    return {"title": "General Support", "solution": "No specific article found."}

def check_system_status() -> dict:
    """Check current status of all platform services."""
    return {"overall": "operational", "auth_service": "degraded", "last_incident": "2026-02-08"}

technical_react_agent = create_react_agent(
    llm,
    tools=[search_knowledge_base, check_system_status],
    prompt="You are a technical specialist. Use search_knowledge_base and check_system_status.",
)

def technical_node(state: MessagesState):
    result = technical_react_agent.invoke(state)
    return {"messages": result["messages"][len(state["messages"]):]}

# --- Layer 2: Billing Agent (MCP -> Supabase) ---
# ADK equivalent: Agent(name="billing_agent_mcp", tools=[supabase_mcp])

def _build_mcp_config() -> dict:
    args = ["-y", "@supabase/mcp-server-supabase@latest", "--access-token", TOKEN]
    if PROJECT_REF:
        args += ["--project-ref", PROJECT_REF]
    return {"supabase": {"command": "npx", "args": args, "transport": "stdio"}}

async def billing_node(state: MessagesState):
    client = MultiServerMCPClient(_build_mcp_config())
    tools = await client.get_tools()
    agent = create_react_agent(
        llm,
        tools=tools,
        prompt="You are a billing specialist. Use MCP tools to query customers, orders, support_tickets.",
    )
    result = await agent.ainvoke(state)
    return {"messages": result["messages"][len(state["messages"]):]}

# --- Layer 3: Shipping Node (HTTP -> remote agent service) ---
# ADK equivalent: RemoteA2aAgent(name="shipping_agent", agent_card="http://localhost:8001")
# LangGraph equivalent: HTTP POST to the shipping_agent FastAPI service

async def shipping_node(state: MessagesState):
    user_message = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "http://localhost:8001/invoke",
                json={"message": user_message},
            )
            response_text = resp.json()["response"]
    except Exception as e:
        response_text = f"Shipping agent unavailable: {e}"
    return {"messages": [AIMessage(content=response_text, name="shipping_agent")]}

# --- Root Supervisor ---
# ADK equivalent: Agent(name="full_support_system", sub_agents=[...])

class Router(BaseModel):
    """Routing decision for the supervisor."""
    next: Literal["billing_agent", "technical_agent", "shipping_agent", "FINISH"]

SUPERVISOR_PROMPT = (
    "You are a customer support router. Route to:\n"
    "- billing_agent: billing, invoices, payments (has live database access)\n"
    "- technical_agent: bugs, crashes, performance, system status\n"
    "- shipping_agent: package tracking, delivery status, order shipping\n"
    "Output FINISH when the conversation is already resolved."
)

def supervisor_node(state: MessagesState) -> Command[Literal["billing_agent", "technical_agent", "shipping_agent", "__end__"]]:
    response = llm.with_structured_output(Router).invoke(
        [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]
    )
    goto = "__end__" if response.next == "FINISH" else response.next
    return Command(goto=goto)

# --- Graph ---

builder = StateGraph(MessagesState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("billing_agent", billing_node)
builder.add_node("technical_agent", technical_node)
builder.add_node("shipping_agent", shipping_node)

builder.add_edge(START, "supervisor")
builder.add_edge("billing_agent", END)
builder.add_edge("technical_agent", END)
builder.add_edge("shipping_agent", END)

app = builder.compile()

# --- Runner ---

async def ask(message: str) -> str:
    langfuse_handler = LangfuseHandler()
    result = await app.ainvoke(
        {"messages": [HumanMessage(content=message)]},
        config={"callbacks": [langfuse_handler]},
    )
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            return msg.content
    return "(no response)"

async def main():
    scenarios = [
        ("BILLING (MCP)", "I'm Jane Doe (jane@example.com). What plan am I on? Show my recent orders."),
        ("TECHNICAL (Local)", "My app is really slow lately. Is something wrong with your servers?"),
        ("SHIPPING (Remote)", "Where is my package for order ORD-1004? When will it arrive?"),
    ]
    for label, query in scenarios:
        print(f"\n--- {label} ---")
        print(f"User: {query}\n")
        print(f"Agent: {await ask(query)}\n")

if __name__ == "__main__":
    asyncio.run(main())
