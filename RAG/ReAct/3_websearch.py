"""
ReAct Agent with Web Search (DuckDuckGo) using LangChain

- Loads your OPENAI_API_KEY from .env
- Defines a @tool `web_search` (DuckDuckGo, no API key needed)
- Uses the standard ReAct prompt from LangChain Hub ("hwchase17/react")
- Creates an agent + executor and runs a sample query

Setup:
  pip install -U langchain langchain-openai python-dotenv duckduckgo-search
  # optional if you haven't used the hub before:
  pip install -U langchain-community

.env (in same folder):
  OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
"""

from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv

# LangChain Hub (for the stock ReAct prompt)
from langchain import hub

# Agent building blocks
from langchain.agents import create_react_agent, AgentExecutor

# NOTE: The @tool decorator moved to langchain.tools in newer versions.
# Many tutorials still import from langchain.agents. We support both:
try:
    from langchain.tools import tool  # preferred, newer
except Exception:
    from langchain.agents import tool  # fallback for older versions

# OpenAI chat model wrapper
from langchain_openai import ChatOpenAI

# Lightweight web search (no API key required)
from duckduckgo_search import DDGS


# ------------------------------------------------------------
# STEP 1: Load environment variables
# ------------------------------------------------------------
load_dotenv()  # reads .env so ChatOpenAI can pick up OPENAI_API_KEY


# ------------------------------------------------------------
# STEP 2: Define a custom TOOL
# ------------------------------------------------------------
@tool
def web_search(query: str, max_results: int = 3) -> str:
    """
    Search the web using DuckDuckGo and return the top results.

    Parameters:
      query (str): The user's search query.
      max_results (int): Number of top results to return (default=3).

    Returns:
      str: A formatted string containing titles, links and snippets.

    Notes:
      - Decorated with @tool so the agent can call it when it needs real-time info.
      - Uses DDGS().text(...) which returns [{'title','href','body'}, ...]
    """
    if not query or not query.strip():
        return "No query provided."

    rows: List[str] = []
    with DDGS() as ddg:
        results = ddg.text(query, max_results=max_results)
        for i, r in enumerate(results or [], start=1):
            title = r.get("title") or "No title"
            link = r.get("href") or "No link"
            desc = r.get("body") or ""
            rows.append(f"{i}. {title}\n   {link}\n   {desc}")

    return "\n".join(rows) if rows else "No results found."


# ------------------------------------------------------------
# STEP 3: Initialize the LLM (Language Model)
# ------------------------------------------------------------
# Choose a fast, cost-effective model you have access to.
# Examples: "gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ------------------------------------------------------------
# STEP 4: Define Tools and Prompt
# ------------------------------------------------------------
tools = [web_search]

# Stock ReAct prompt template from the LangChain Hub
# Contains the Thought/Action/Observation scaffolding
prompt = hub.pull("hwchase17/react")


# ------------------------------------------------------------
# STEP 5: Create the Agent
# ------------------------------------------------------------
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# ------------------------------------------------------------
# STEP 6: Run the Agent
# ------------------------------------------------------------
if __name__ == "__main__":
    # Change this query to anything—the agent will decide to call web_search as needed.
    query = "Who is the current Prime Minister of Canada?"

    # You can also accept user input from CLI:
    # import sys
    # if len(sys.argv) > 1:
    #     query = " ".join(sys.argv[1:])

    result = agent_executor.invoke({"input": query})

    print("\n✅ FINAL ANSWER:")
    # In recent LangChain, the AgentExecutor returns a dict with key "output"
    print(result.get("output", result))
