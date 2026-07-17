"""
Demo 2: MCP -- Agent with Real Database Access (Supabase)

ADK equivalent: McpToolset(connection_params=StdioConnectionParams(...))
LangGraph equivalent: langchain_mcp_adapters.client.MultiServerMCPClient
                      loads MCP tools into a create_react_agent.

Run: python demo2_mcp.py
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TOKEN = os.getenv("SUPABASE_ACCESS_TOKEN", "")
PROJECT_REF = os.getenv("SUPABASE_PROJECT_REF", "")

if not TOKEN:
    sys.exit("Set SUPABASE_ACCESS_TOKEN in .env (https://supabase.com/dashboard/account/tokens)")

llm = ChatOpenAI(model=MODEL, temperature=0)

# --- MCP Client (launches Supabase MCP server as subprocess) ---
# ADK equivalent: McpToolset(connection_params=StdioConnectionParams(...))
# LangGraph equivalent: MultiServerMCPClient with stdio transport

def _build_mcp_config() -> dict:
    args = ["-y", "@supabase/mcp-server-supabase@latest", "--access-token", TOKEN]
    if PROJECT_REF:
        args += ["--project-ref", PROJECT_REF]
    return {
        "supabase": {
            "command": "npx",
            "args": args,
            "transport": "stdio",
        }
    }

# --- Agent ---
# ADK equivalent: Agent(name=..., tools=[supabase_mcp])
# LangGraph equivalent: create_react_agent(llm, tools=client.get_tools())

async def ask(message: str) -> str:
    client = MultiServerMCPClient(_build_mcp_config())
    tools = await client.get_tools()
    agent = create_react_agent(
        llm,
        tools=tools,
        prompt=(
            "You are a billing specialist with real database access. "
            "Use the available MCP tools to query customers, orders, and support_tickets tables. "
            "Always look up the customer first before answering."
        ),
    )
    result = await agent.ainvoke({"messages": [HumanMessage(content=message)]})
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            return msg.content
    return "(no response)"

async def main():
    tests = [
        ("CUSTOMER LOOKUP", "What orders does Bob Smith have? What's the total amount?"),
        ("CROSS-TABLE QUERY", "Show me all high-priority open support tickets with customer name and email."),
    ]
    for label, query in tests:
        print(f"\n--- {label} ---")
        print(f"User: {query}\n")
        print(f"Agent: {await ask(query)}\n")

if __name__ == "__main__":
    asyncio.run(main())
