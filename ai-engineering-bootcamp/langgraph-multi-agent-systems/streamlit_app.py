"""
LangGraph Multi-Agent Systems -- Interactive Classroom Demo

Run:
    uvicorn shipping_agent:app --port 8001   # Terminal 1
    streamlit run streamlit_app.py            # Terminal 2
"""

import streamlit as st
import asyncio
import concurrent.futures
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from langfuse.langchain import CallbackHandler as LangfuseHandler
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from pydantic import BaseModel
from typing import Literal
import httpx

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

def create_escalation_ticket(customer_email: str, issue_summary: str, priority: str) -> dict:
    """Create an escalation ticket for issues needing human review."""
    times = {"low": "48 hours", "medium": "24 hours", "high": "4 hours", "critical": "1 hour"}
    return {"ticket_id": "ESC-2026-0042", "priority": priority, "estimated_response": times.get(priority, "24 hours")}

# --- Supervisor Router ---

class Router(BaseModel):
    next: Literal["billing_agent", "technical_agent", "escalation_agent", "FINISH"]

class FullRouter(BaseModel):
    next: Literal["billing_agent", "technical_agent", "shipping_agent", "FINISH"]

SUPERVISOR_PROMPT = (
    "You are a customer support router. Route to:\n"
    "- billing_agent: invoices, payments, refunds\n"
    "- technical_agent: bugs, crashes, performance, system status\n"
    "- escalation_agent: complaints, disputes, fraud, security\n"
    "Output FINISH when resolved."
)

FULL_SUPERVISOR_PROMPT = (
    "You are a customer support router. Route to:\n"
    "- billing_agent: billing, invoices, payments (has live database access)\n"
    "- technical_agent: bugs, crashes, performance, system status\n"
    "- shipping_agent: package tracking, delivery status\n"
    "Output FINISH when resolved."
)

def _make_supervisor(router_model, prompt, options):
    def supervisor_node(state: MessagesState) -> Command:
        response = llm.with_structured_output(router_model).invoke(
            [SystemMessage(content=prompt)] + state["messages"]
        )
        goto = "__end__" if response.next == "FINISH" else response.next
        note = "Conversation finished." if goto == "__end__" else f"Routing to {goto}"
        return Command(
            goto=goto,
            update={"messages": [AIMessage(content=note, name="supervisor")]},
        )
    return supervisor_node

def _make_specialist(tools, system_prompt):
    agent = create_react_agent(llm, tools=tools, prompt=system_prompt)
    def node(state: MessagesState):
        result = agent.invoke(state)
        return {"messages": result["messages"][len(state["messages"]):]}
    return node

# --- Demo 1: Routing Graph ---

def build_demo1_graph():
    billing_node = _make_specialist([lookup_invoice, process_refund], "You are a billing specialist. Use lookup_invoice and process_refund.")
    technical_node = _make_specialist([search_knowledge_base, check_system_status], "You are a technical specialist. Use search_knowledge_base and check_system_status.")
    escalation_node = _make_specialist([create_escalation_ticket], "You are an escalation specialist. Use create_escalation_ticket to log cases.")
    supervisor = _make_supervisor(Router, SUPERVISOR_PROMPT, ["billing_agent", "technical_agent", "escalation_agent"])

    builder = StateGraph(MessagesState)
    builder.add_node("supervisor", supervisor)
    builder.add_node("billing_agent", billing_node)
    builder.add_node("technical_agent", technical_node)
    builder.add_node("escalation_agent", escalation_node)
    builder.add_edge(START, "supervisor")
    builder.add_edge("billing_agent", END)
    builder.add_edge("technical_agent", END)
    builder.add_edge("escalation_agent", END)
    return builder.compile()

# --- Demo 2: MCP Graph ---

async def build_demo2_agent():
    from langchain_mcp_adapters.client import MultiServerMCPClient
    token = os.getenv("SUPABASE_ACCESS_TOKEN", "")
    ref = os.getenv("SUPABASE_PROJECT_REF", "")
    if not token:
        return None, "SUPABASE_ACCESS_TOKEN not set in .env"
    args = ["-y", "@supabase/mcp-server-supabase@latest", "--access-token", token]
    if ref:
        args += ["--project-ref", ref]
    config = {"supabase": {"command": "npx", "args": args, "transport": "stdio"}}
    return config, None

# --- Demo 3: Full System Graph ---

def build_demo3_graph(token, ref):
    from langchain_mcp_adapters.client import MultiServerMCPClient

    args = ["-y", "@supabase/mcp-server-supabase@latest", "--access-token", token]
    if ref:
        args += ["--project-ref", ref]
    mcp_config = {"supabase": {"command": "npx", "args": args, "transport": "stdio"}}

    technical_react = create_react_agent(
        llm,
        tools=[search_knowledge_base, check_system_status],
        prompt="You are a technical specialist. Use search_knowledge_base and check_system_status.",
    )

    def technical_node(state: MessagesState):
        result = technical_react.invoke(state)
        return {"messages": result["messages"][len(state["messages"]):]}

    async def billing_node(state: MessagesState):
        client = MultiServerMCPClient(mcp_config)
        tools = await client.get_tools()
        agent = create_react_agent(
            llm, tools=tools,
            prompt="You are a billing specialist. Use MCP tools to query customers, orders, support_tickets.",
        )
        result = await agent.ainvoke(state)
        return {"messages": result["messages"][len(state["messages"]):]}

    async def shipping_node(state: MessagesState):
        user_message = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), ""
        )
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post("http://localhost:8001/invoke", json={"message": user_message})
                data = resp.json()
                text = data["response"]
                remote_trace = data.get("trace") or []
        except Exception as e:
            text = f"Shipping agent unavailable: {e}"
            remote_trace = []

        # Rebuild remote tool steps so Agent Trace shows them in the UI
        new_messages = []
        pending_tool_call_id = None
        for i, step in enumerate(remote_trace):
            step_type = step.get("type")
            if step_type == "tool_call":
                pending_tool_call_id = f"ship_tc_{i}"
                new_messages.append(AIMessage(
                    content="",
                    name="shipping_agent",
                    tool_calls=[{
                        "id": pending_tool_call_id,
                        "name": step.get("tool") or "tool",
                        "args": step.get("args") or {},
                    }],
                ))
            elif step_type == "tool_response":
                new_messages.append(ToolMessage(
                    content=step.get("result") or "",
                    name=step.get("tool") or "tool",
                    tool_call_id=pending_tool_call_id or f"ship_tc_{i}",
                ))
            elif step_type == "text" and step.get("text"):
                new_messages.append(AIMessage(content=step["text"], name="shipping_agent"))
        if not new_messages:
            new_messages = [AIMessage(content=text, name="shipping_agent")]
        return {"messages": new_messages}

    def supervisor_node(state: MessagesState) -> Command:
        response = llm.with_structured_output(FullRouter).invoke(
            [SystemMessage(content=FULL_SUPERVISOR_PROMPT)] + state["messages"]
        )
        goto = "__end__" if response.next == "FINISH" else response.next
        note = "Conversation finished." if goto == "__end__" else f"Routing to {goto}"
        return Command(
            goto=goto,
            update={"messages": [AIMessage(content=note, name="supervisor")]},
        )

    builder = StateGraph(MessagesState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("billing_agent", billing_node)
    builder.add_node("technical_agent", technical_node)
    builder.add_node("shipping_agent", shipping_node)
    builder.add_edge(START, "supervisor")
    builder.add_edge("billing_agent", END)
    builder.add_edge("technical_agent", END)
    builder.add_edge("shipping_agent", END)
    return builder.compile()

# --- Runner ---

def run_graph_sync(graph_or_fn, message: str, timeout=120, is_async_factory=False):
    """Run a LangGraph app synchronously and return (response, trace)."""
    langfuse_handler = LangfuseHandler()

    async def _run():
        if is_async_factory:
            graph = await graph_or_fn()
        else:
            graph = graph_or_fn
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=message)]},
            config={"callbacks": [langfuse_handler]},
        )
        messages = result["messages"]
        final = "(no response)"
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
                final = msg.content
                break
        trace = _extract_trace(messages)
        return final, trace

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, _run()).result(timeout=timeout)

async def run_mcp_query(message: str, mcp_config: dict):
    """Run a single MCP query and return (response, trace)."""
    from langchain_mcp_adapters.client import MultiServerMCPClient
    langfuse_handler = LangfuseHandler()
    client = MultiServerMCPClient(mcp_config)
    tools = await client.get_tools()
    agent = create_react_agent(
        llm, tools=tools,
        prompt="You are a billing specialist with database access. Use MCP tools to query customers, orders, support_tickets.",
    )
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=message)]},
        config={"callbacks": [langfuse_handler]},
    )
    messages = result["messages"]
    final = "(no response)"
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            final = msg.content
            break
    return final, _extract_trace(messages)

def run_mcp_sync(message: str, mcp_config: dict, timeout=180):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, run_mcp_query(message, mcp_config)).result(timeout=timeout)

# --- Trace Extraction ---

def _extract_trace(messages: list) -> list:
    """Convert LangGraph message list into a trace structure for display."""
    trace = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            continue
        elif isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", None)
            name = getattr(msg, "name", None) or "agent"
            if tool_calls:
                for tc in tool_calls:
                    trace.append({"author": name, "type": "tool_call", "tool": tc["name"], "args": tc.get("args", {})})
            elif msg.content:
                trace.append({"author": name, "type": "text", "text": msg.content})
        elif isinstance(msg, ToolMessage):
            trace.append({"author": "tools", "type": "tool_response", "tool": msg.name or "tool", "result": str(msg.content)[:800]})
    return trace

# --- Helpers ---

def check_shipping_agent(url="http://localhost:8001"):
    try:
        import urllib.request
        with urllib.request.urlopen(f"{url}/.well-known/agent-card.json", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False

def render_trace(trace: list):
    if not trace:
        return
    seen = []
    for i, step in enumerate(trace):
        author = step.get("author", "unknown")
        if author not in seen:
            seen.append(author)
            if len(seen) > 1:
                st.info(f"Routed to **{author}**")
        if step["type"] == "tool_call":
            args = ", ".join(f"{k}={v!r}" for k, v in step.get("args", {}).items())
            st.warning(f"**{i+1}.** `{author}` called **{step['tool']}**({args[:200]})")
        elif step["type"] == "tool_response":
            st.success(f"**{i+1}.** `{author}` got result from **{step['tool']}**")
            if step.get("result"):
                st.code(step["result"][:500], language="json")
        elif step["type"] == "text" and step.get("text", "").strip():
            st.markdown(f"**{i+1}.** `{author}`: {step['text'][:300]}")

# --- Page Config ---

st.set_page_config(page_title="LangGraph Multi-Agent Systems", layout="wide")
st.markdown("<style>.block-container{padding-top:1.5rem;}</style>", unsafe_allow_html=True)

api_key = os.getenv("OPENAI_API_KEY")
supa_token = os.getenv("SUPABASE_ACCESS_TOKEN")
supa_ref = os.getenv("SUPABASE_PROJECT_REF")
shipping_ok = check_shipping_agent()

# --- Sidebar ---

with st.sidebar:
    st.title("LangGraph Multi-Agent Systems")
    st.caption("Interactive Classroom Demo")
    page = st.radio("Navigate", ["Overview", "Demo 1: Routing", "Demo 2: MCP + Database", "Demo 3: Full System"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### Status")
    if api_key:
        st.success("OpenAI API Key")
    else:
        st.error("OpenAI API Key -- not set")
    if supa_token and supa_ref:
        st.success("Supabase")
    else:
        st.warning("Supabase -- not configured")
    if shipping_ok:
        st.success("Shipping Agent (:8001)")
    else:
        st.warning("Shipping Agent -- not running")

# --- Overview ---

if page == "Overview":
    st.header("Building Multi-Agent AI Systems with LangGraph")
    st.markdown("Three progressive demos showing multi-agent system design using LangGraph — the LangChain equivalent of Google ADK.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Demo 1: Routing")
        st.markdown("Supervisor node delegates to billing, technical, and escalation specialists.\n\n**ADK:** `Agent(sub_agents=[...])`\n**LangGraph:** `StateGraph` + `Command` routing")
    with col2:
        st.subheader("Demo 2: MCP")
        st.markdown("Agent connects to a live Supabase database via MCP.\n\n**ADK:** `McpToolset`\n**LangGraph:** `MultiServerMCPClient`")
    with col3:
        st.subheader("Demo 3: Full System")
        st.markdown("Combines routing + MCP + remote agent over HTTP.\n\n**ADK:** `RemoteA2aAgent`\n**LangGraph:** HTTP call to FastAPI service")

    st.markdown("---")
    st.subheader("ADK → LangGraph Mapping")
    st.markdown(
        "| ADK Concept | LangGraph Equivalent |\n|---|---|\n"
        "| `Agent(sub_agents=[...])` | `StateGraph` + supervisor `Command` routing |\n"
        "| `McpToolset` | `MultiServerMCPClient` from `langchain-mcp-adapters` |\n"
        "| `RemoteA2aAgent` | HTTP call to a remote FastAPI / LangGraph service |\n"
        "| `Runner` + `InMemorySessionService` | `graph.invoke()` / `graph.ainvoke()` |\n"
        "| `Agent(tools=[...])` | `create_react_agent(llm, tools=[...])` |"
    )

    st.markdown("---")
    st.subheader("Architecture")
    st.code("""
    User Query
        |
        v
    +-----------------------+
    |    Supervisor Node    |   (structured LLM output -> Command routing)
    +---+-------+-------+--+
        |       |       |
        v       v       v
    +------+ +------+ +--------+
    |Billing| | Tech | |Shipping|
    +---+--+ +--+---+ +---+----+
        |       |          |
        v       v          v
    +------+ +------+ +--------+
    | MCP  | |Local | |  HTTP  |
    |Client| |Tools | |  POST  |
    +---+--+ +------+ +---+----+
        |                  |
        v                  v
    +------+          +--------+
    |Supa- |          | FastAPI|
    | base |          | :8001  |
    +------+          +--------+
    """, language=None)

    st.markdown("---")
    st.subheader("Multi-Agent Patterns in LangGraph")
    for name, desc in {
        "Supervisor / Delegation": "Supervisor node uses `llm.with_structured_output(Router)` to pick the next node via `Command(goto=...)`.",
        "Sequential Pipeline": "Chain nodes with `add_edge`: A → B → C → END. Each node processes and passes state forward.",
        "Parallel Fan-Out": "Use `Send` to dispatch to multiple nodes simultaneously. Collect results in a reducer.",
        "Loop / Iterative Refinement": "Add a conditional edge that loops back to a node until a quality condition is met.",
    }.items():
        with st.expander(name):
            st.markdown(desc)

# --- Demo 1 ---

elif page == "Demo 1: Routing":
    st.header("Demo 1: Multi-Agent Routing")
    st.markdown("A **supervisor node** uses structured LLM output to route to the right specialist.")

    with st.expander("Architecture"):
        st.code("""
    Supervisor (Router LLM)
      |--- billing_agent   -> lookup_invoice, process_refund
      |--- technical_agent  -> search_knowledge_base, check_system_status
      |--- escalation_agent -> create_escalation_ticket
        """, language=None)
    with st.expander("ADK vs LangGraph"):
        st.code("""
# ADK
root_agent = Agent(sub_agents=[billing_agent, technical_agent, escalation_agent])

# LangGraph
def supervisor_node(state) -> Command:
    response = llm.with_structured_output(Router).invoke(state["messages"])
    return Command(goto=response.next)
        """, language="python")

    if not api_key:
        st.error("Set OPENAI_API_KEY in .env"); st.stop()

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    query = None
    with col1:
        if st.button("Billing Query", use_container_width=True):
            query = "I need to check my invoice. Email: bob@example.com. Can I get a refund?"
    with col2:
        if st.button("Technical Query", use_container_width=True):
            query = "My app keeps crashing when I login. Is there an outage?"
    with col3:
        if st.button("Escalation Query", use_container_width=True):
            query = "Someone hacked my account! Email: jane@example.com. Urgent!"
    custom = st.text_input("Or type your own:", placeholder="e.g. What's the status of my refund?")
    if st.button("Send", type="primary") and custom.strip():
        query = custom.strip()

    if query:
        st.markdown(f"**Query:** {query}")
        with st.spinner("Running..."):
            try:
                graph = build_demo1_graph()
                response, trace = run_graph_sync(graph, query)
                st.session_state["d1_resp"], st.session_state["d1_trace"], st.session_state["d1_q"] = response, trace, query
            except Exception as e:
                st.error(str(e))

    if st.session_state.get("d1_resp"):
        st.markdown("---")
        st.markdown(f"**Query:** {st.session_state.get('d1_q', '')}")
        st.subheader("Response")
        st.markdown(st.session_state["d1_resp"])
        with st.expander("Agent Trace", expanded=True):
            render_trace(st.session_state.get("d1_trace", []))

# --- Demo 2 ---

elif page == "Demo 2: MCP + Database":
    st.header("Demo 2: MCP -- Real Database Access")
    st.markdown("The agent connects to a **live Supabase database** via MCP using `langchain-mcp-adapters`.")

    with st.expander("Architecture"):
        st.code("create_react_agent -> MultiServerMCPClient -> Supabase MCP Server (npx) -> Supabase DB", language=None)
    with st.expander("ADK vs LangGraph"):
        st.code("""
# ADK
supabase_mcp = McpToolset(connection_params=StdioConnectionParams(...))
agent = Agent(tools=[supabase_mcp])

# LangGraph (v0.1.0+: no context manager)
client = MultiServerMCPClient(config)
tools = await client.get_tools()
agent = create_react_agent(llm, tools=tools)
        """, language="python")

    if not api_key:
        st.error("Set OPENAI_API_KEY in .env"); st.stop()
    if not supa_token:
        st.error("Set SUPABASE_ACCESS_TOKEN and SUPABASE_PROJECT_REF in .env"); st.stop()

    st.markdown("---")
    col1, col2 = st.columns(2)
    query = None
    with col1:
        if st.button("Customer Lookup", use_container_width=True):
            query = "What orders does Bob Smith have? What's the total?"
    with col2:
        if st.button("Cross-Table Query", use_container_width=True):
            query = "Show all high-priority open support tickets with customer name."
    custom = st.text_input("Or type your own:", key="d2c", placeholder="e.g. How many customers are on the pro plan?")
    if st.button("Send", key="d2s", type="primary") and custom.strip():
        query = custom.strip()

    if query:
        st.markdown(f"**Query:** {query}")
        with st.spinner("Connecting to MCP + Supabase (may take 10-15s)..."):
            try:
                args = ["-y", "@supabase/mcp-server-supabase@latest", "--access-token", supa_token]
                if supa_ref:
                    args += ["--project-ref", supa_ref]
                mcp_config = {"supabase": {"command": "npx", "args": args, "transport": "stdio"}}
                response, trace = run_mcp_sync(query, mcp_config, timeout=180)
                st.session_state["d2_resp"], st.session_state["d2_trace"], st.session_state["d2_q"] = response, trace, query
            except Exception as e:
                st.error(str(e))

    if st.session_state.get("d2_resp"):
        st.markdown("---")
        st.markdown(f"**Query:** {st.session_state.get('d2_q', '')}")
        st.subheader("Response")
        st.markdown(st.session_state["d2_resp"])
        with st.expander("Agent Trace", expanded=True):
            render_trace(st.session_state.get("d2_trace", []))

# --- Demo 3 ---

elif page == "Demo 3: Full System":
    st.header("Demo 3: Full System -- Routing + MCP + Remote Agent")
    st.markdown("Combines **routing** + **MCP** (Supabase) + **remote agent** (HTTP call to shipping service).")

    with st.expander("Architecture"):
        st.code("""
    Supervisor -> billing_agent  (MCP -> Supabase)
               -> technical_agent (local tools)
               -> shipping_agent  (HTTP POST -> localhost:8001)
        """, language=None)
    with st.expander("ADK vs LangGraph"):
        st.code("""
# ADK remote agent
shipping = RemoteA2aAgent(name="shipping_agent", agent_card="http://localhost:8001")

# LangGraph remote agent
async def shipping_node(state):
    async with httpx.AsyncClient() as client:
        resp = await client.post("http://localhost:8001/invoke", json={"message": query})
    return {"messages": [AIMessage(content=resp.json()["response"])]}
        """, language="python")

    if not api_key:
        st.error("Set OPENAI_API_KEY in .env"); st.stop()
    if not supa_token:
        st.warning("Supabase not configured -- billing won't work.")
    if not shipping_ok:
        st.warning("Shipping agent not running. Start: `uvicorn shipping_agent:app --port 8001`")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    query = None
    with col1:
        if st.button("Billing (MCP)", use_container_width=True):
            query = "I'm Jane Doe (jane@example.com). What plan am I on?"
    with col2:
        if st.button("Technical (Local)", use_container_width=True):
            query = "My app is slow. Is something wrong with your servers?"
    with col3:
        if st.button("Shipping (Remote)", use_container_width=True):
            query = "Where is my package for order ORD-1004?"
    custom = st.text_input("Or type your own:", key="d3c", placeholder="e.g. Help with my order and a tech issue")
    if st.button("Send", key="d3s", type="primary") and custom.strip():
        query = custom.strip()

    if query:
        st.markdown(f"**Query:** {query}")
        with st.spinner("Running full system (15-20s)..."):
            try:
                if not supa_token:
                    st.error("SUPABASE_ACCESS_TOKEN required for full system demo.")
                else:
                    graph = build_demo3_graph(supa_token, supa_ref)
                    response, trace = run_graph_sync(graph, query, timeout=180)
                    st.session_state["d3_resp"], st.session_state["d3_trace"], st.session_state["d3_q"] = response, trace, query
            except Exception as e:
                st.error(str(e))

    if st.session_state.get("d3_resp"):
        st.markdown("---")
        st.markdown(f"**Query:** {st.session_state.get('d3_q', '')}")
        st.subheader("Response")
        st.markdown(st.session_state["d3_resp"])
        with st.expander("Agent Trace", expanded=True):
            render_trace(st.session_state.get("d3_trace", []))
