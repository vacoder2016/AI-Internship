"""
Demo 1: Multi-Agent Routing with LangGraph

ADK equivalent: Agent with sub_agents= (router delegating to specialists)
LangGraph pattern: Supervisor node uses structured output to route;
                   specialist nodes are create_react_agent instances.

Run: python demo1_routing.py
"""

import os
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(model=MODEL, temperature=0)

# --- Tools ---

def lookup_invoice(customer_email: str) -> dict:
    """Look up the most recent invoice for a customer by email."""
    invoices = {
        "bob@example.com": {"invoice_id": "INV-2024-002", "customer": "Bob Smith", "amount": "$299.00", "status": "overdue", "plan": "Enterprise Annual"},
        "jane@example.com": {"invoice_id": "INV-2024-001", "customer": "Jane Doe", "amount": "$99.00", "status": "paid", "plan": "Pro Monthly"},
        "alice@example.com": {"invoice_id": "INV-2024-003", "customer": "Alice Johnson", "amount": "$49.00", "status": "paid", "plan": "Starter Monthly"},
    }
    return invoices.get(customer_email.lower(), {"error": f"No invoice found for {customer_email}"})

def process_refund(invoice_id: str, reason: str) -> dict:
    """Process a refund for a specific invoice."""
    return {"refund_id": f"REF-{invoice_id[-3:]}", "status": "approved", "message": f"Refund for {invoice_id} approved. 5-7 business days."}

def search_knowledge_base(query: str) -> dict:
    """Search the knowledge base for technical solutions."""
    articles = {
        "login": {"title": "Login Issues", "solution": "1. Clear cache. 2. Try incognito. 3. Reset password. 4. 3 failed attempts = 30 min lockout."},
        "crash": {"title": "App Crashing", "solution": "1. Update to v3.2.1. 2. Clear app data. 3. Check OS: iOS 16+ / Android 13+."},
        "slow": {"title": "Performance Issues", "solution": "1. Check internet (min 5 Mbps). 2. Close other apps. 3. Enable hardware acceleration."},
    }
    for keyword, article in articles.items():
        if keyword in query.lower():
            return article
    return {"title": "General Support", "solution": "No specific article found. Contact support."}

def check_system_status() -> dict:
    """Check current status of all platform services."""
    return {
        "overall": "operational",
        "services": {"web_app": "operational", "api": "operational", "database": "operational", "auth_service": "degraded"},
        "last_incident": "2026-02-08 -- Auth service brief outage (resolved)",
    }

def create_escalation_ticket(customer_email: str, issue_summary: str, priority: str) -> dict:
    """Create an escalation ticket for issues needing human review."""
    times = {"low": "48 hours", "medium": "24 hours", "high": "4 hours", "critical": "1 hour"}
    return {"ticket_id": "ESC-2026-0042", "priority": priority, "estimated_response": times.get(priority, "24 hours")}

# --- Supervisor Router ---
# ADK equivalent: root_agent with sub_agents decides routing automatically
# LangGraph equivalent: supervisor node uses structured LLM output to pick the next node

class Router(BaseModel):
    """Routing decision for the supervisor."""
    next: Literal["billing_agent", "technical_agent", "escalation_agent", "FINISH"]

SUPERVISOR_PROMPT = (
    "You are a customer support router. Decide which specialist should handle this request:\n"
    "- billing_agent: invoices, payments, refunds, billing questions\n"
    "- technical_agent: bugs, crashes, performance issues, system status\n"
    "- escalation_agent: complaints, disputes, fraud, security concerns\n"
    "Output FINISH when the conversation is already resolved."
)

def supervisor_node(state: MessagesState) -> Command[Literal["billing_agent", "technical_agent", "escalation_agent", "__end__"]]:
    response = llm.with_structured_output(Router).invoke(
        [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]
    )
    goto = "__end__" if response.next == "FINISH" else response.next
    return Command(goto=goto)

# --- Specialist Agent Nodes ---
# ADK equivalent: Agent(name=..., tools=[...]) assigned as sub_agent
# LangGraph equivalent: create_react_agent wrapped in a graph node

def _make_specialist(tools: list, system_prompt: str):
    """Create a specialist react agent and wrap it as a graph node."""
    agent = create_react_agent(
        llm,
        tools=tools,
        prompt=system_prompt,
    )
    def node(state: MessagesState):
        result = agent.invoke(state)
        new_messages = result["messages"][len(state["messages"]):]
        return {"messages": new_messages}
    return node

billing_node = _make_specialist(
    tools=[lookup_invoice, process_refund],
    system_prompt="You are a billing specialist. Use lookup_invoice to find invoices and process_refund for refunds.",
)
technical_node = _make_specialist(
    tools=[search_knowledge_base, check_system_status],
    system_prompt="You are a technical specialist. Use search_knowledge_base and check_system_status to help users.",
)
escalation_node = _make_specialist(
    tools=[create_escalation_ticket],
    system_prompt="You are an escalation specialist. Use create_escalation_ticket to log cases requiring human review.",
)

# --- Graph ---
# ADK equivalent: Runner(agent=root_agent, ...)
# LangGraph equivalent: StateGraph compiled to an app

builder = StateGraph(MessagesState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("billing_agent", billing_node)
builder.add_node("technical_agent", technical_node)
builder.add_node("escalation_agent", escalation_node)

builder.add_edge(START, "supervisor")
builder.add_edge("billing_agent", END)
builder.add_edge("technical_agent", END)
builder.add_edge("escalation_agent", END)

app = builder.compile()

# --- Runner ---

def ask(message: str) -> str:
    result = app.invoke({"messages": [HumanMessage(content=message)]})
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            return msg.content
    return "(no response)"

if __name__ == "__main__":
    tests = [
        ("BILLING", "I need to check my latest invoice. My email is bob@example.com. Can I get a refund?"),
        ("TECHNICAL", "My app keeps crashing every time I try to login. Is there an outage?"),
        ("ESCALATION", "Someone hacked my account! I see charges I didn't make. Email: jane@example.com. Urgent!"),
    ]
    for label, query in tests:
        print(f"\n--- {label} ---")
        print(f"User: {query}\n")
        print(f"Agent: {ask(query)}\n")
