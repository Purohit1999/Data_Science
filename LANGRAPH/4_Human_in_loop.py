"""
graph_combined_explained.py

This script builds a LangGraph chatbot that:
 - Uses an LLM (via langchain) and binds tools to it (Tavily web search + a human-assistance tool)
 - Uses an InMemorySaver as a checkpointer so conversation threads can persist within the process
 - Compiles the graph, visualizes it (Mermaid PNG), and demonstrates running it once and in a loop

Important: keep OPENAI_API_KEY and TAVILY_API_KEY in your .env before running.
"""

# ---------------------------
# Imports & environment
# ---------------------------
from typing import Annotated, List
from typing_extensions import TypedDict

# LangChain chat model initializer (wraps provider-specific LLMs)
from langchain.chat_models import init_chat_model

# TavilySearch: a web-search tool wrapper for LangChain
from langchain_tavily import TavilySearch

# tool decorator for creating LangChain-style tool callables
from langchain_core.tools import tool

# LangGraph checkpointer (simple in-memory saver used for demos)
from langgraph.checkpoint.memory import InMemorySaver

# LangGraph core graph builder + a START constant
from langgraph.graph import StateGraph, START

# add_messages annotation instructs LangGraph to append messages instead of replacing
from langgraph.graph.message import add_messages

# Prebuilt ToolNode and helper decision function for conditional routing to tools
from langgraph.prebuilt import ToolNode, tools_condition

# A utility to "interrupt" the runtime to request human assistance (LangGraph helper)
from langgraph.types import interrupt

# Load env variables (OPENAI_API_KEY, TAVILY_API_KEY, etc.) from .env
from dotenv import load_dotenv
load_dotenv()


# ---------------------------
# 1. Define the State schema
# ---------------------------
class State(TypedDict):
    """
    TypedDict defines the expected shape of the runtime `state` that flows through nodes.

    Fields:
      - messages: a list of chat messages.
        We annotate with `add_messages` so that when a node returns {"messages": [...]},
        LangGraph will append that list to the existing state's messages list instead
        of overwriting it.
    """
    messages: Annotated[List, add_messages]


# Create a graph builder that enforces/uses the State type above
graph_builder = StateGraph(State)


# ---------------------------
# 2. Define Tools
# ---------------------------
@tool
def human_assistance(query: str) -> str:
    """
    A tool that requests human assistance for the given query.

    - @tool decorator: registers this function as a tool that an LLM can request.
      The decorated function's signature tells the runtime what input it expects.

    - interrupt({"query": query}): A LangGraph helper that signals an external
      human-in-the-loop system to handle the query. It returns a dict-like response
      (shape depends on the runtime) — here we expect a "data" field with the human reply.

    Returns:
      - The human responder's text (response["data"]) so it can be fed back into the graph.
    """
    # Send an interrupt to the host runtime asking for human help — runtime handles routing.
    response = interrupt({"query": query})
    # Return only the textual data portion to keep message shape simple.
    return response["data"]


# Tavily is our web-search tool. max_results limits how many search results will be returned.
tavily_tool = TavilySearch(max_results=2)

# tools is the list of available tools. We include both the web search tool and the human tool.
tools = [tavily_tool, human_assistance]


# ---------------------------
# 3. Initialize LLM and bind tools
# ---------------------------
# init_chat_model creates a chat model wrapper. Here we specify the model and provider.
# Note: this wrapper will consult OPENAI_API_KEY from environment.
llm = init_chat_model(model="gpt-5-nano", model_provider="openai")

# bind_tools returns a version of the LLM wrapper capable of performing tool-calls:
# if the model decides to call a tool it will be routed to the corresponding function.
llm_with_tools = llm.bind_tools(tools)


# ---------------------------
# 4. Define the chatbot node
# ---------------------------
def chatbot(state: State) -> dict:
    """
    Node function for the "chatbot" node.

    Input:
      - state: current graph state containing "messages" (chat history)

    Action:
      - Calls the LLM (which has tools bound). We pass the current message history.
      - The LLM returns an assistant message object (format may vary by wrapper).

    Return:
      - A dict with keys matching State fields; here {"messages": [assistant_msg]} instructs
        LangGraph to append the assistant_msg to the existing messages because of add_messages.
    """
    assistant_message = llm_with_tools.invoke(state["messages"])
    return {"messages": [assistant_message]}


# Register the chatbot node with the graph builder under the name "chatbot"
graph_builder.add_node("chatbot", chatbot)


# ---------------------------
# 5. ToolNode: a node to execute tools
# ---------------------------
# ToolNode is a prebuilt node that orchestrates calling external tools supplied in a list.
# Here we provide only the Tavily search tool to the ToolNode. (human_assistance is a tool
# the LLM may request directly through tool calling; ToolNode usage may vary by lib version.)
tool_node = ToolNode(tools=[tavily_tool])
graph_builder.add_node("tools", tool_node)

# Add conditional edges: after chatbot runs, tools_condition inspects the LLM output and
# decides whether to route execution to the "tools" node (for example, when the model
# indicates it needs web search/human assistance).
graph_builder.add_conditional_edges("chatbot", tools_condition)

# After the tools node completes, route back to the chatbot so the LLM can produce
# a final integrated answer using tool results.
graph_builder.add_edge("tools", "chatbot")

# Entry point: define what node the graph run should start from. Using "chatbot".
graph_builder.add_edge(START, "chatbot")
graph_builder.set_entry_point("chatbot")


# ---------------------------
# 6. Memory / checkpointer
# ---------------------------
# InMemorySaver stores conversation traces in memory keyed by thread_id. It is ephemeral:
# data is lost when the process exits. For production use a persistent checkpointer.
memory = InMemorySaver()

# Compile the graph into a runnable Graph instance and attach the checkpointer.
# compile() validates the graph and returns the runtime object (.stream(), .invoke(), .get_graph())
graph = graph_builder.compile(checkpointer=memory)


# ---------------------------
# 7. Save graph visual (Mermaid PNG)
# ---------------------------
# draw_mermaid_png returns bytes of a PNG. Save to a file to inspect the flow visually.
png_data = graph.get_graph().draw_mermaid_png()
with open("graph_combined.png", "wb") as f:
    f.write(png_data)
print("✅ Graph saved as graph_combined.png")


# ---------------------------
# 8. Run Example once (single query)
# ---------------------------
# Prepare a single query and thread_id so the checkpointer can load/store the state for "1"
user_input = "I need expert guidance for building an AI agent. Could you request assistance?"
config = {"configurable": {"thread_id": "1"}}

# graph.stream returns an iterator of events/state values while the graph executes.
# stream_mode="values" asks for the raw state dicts as values.
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)

# Iterate events and pretty-print the assistant's latest message when available.
for event in events:
    if "messages" in event:
        # The last message is expected to be the assistant's response. pretty_print is
        # a convenience method offered by some message wrappers; if unavailable, fallback.
        try:
            event["messages"][-1].pretty_print()
        except Exception:
            # Fallback to standard printing
            msg = event["messages"][-1]
            if hasattr(msg, "content"):
                print("Assistant:", msg.content)
            elif isinstance(msg, dict) and "content" in msg:
                print("Assistant:", msg["content"])
            else:
                print("Assistant (raw):", repr(msg))


# ---------------------------
# 9. Interactive chat loop (threaded memory)
# ---------------------------
print("\n🤖 Chatbot with Memory is ready! Type 'quit', 'exit', or 'q' to stop.\n")
while True:
    # Ask the user for a thread id. This is used by the checkpointer so multiple
    # independent conversations can be stored and resumed in the same process.
    thread_id = input("Enter a thread id for this conversation (any string, e.g., '1'): ")
    config = {"configurable": {"thread_id": thread_id}}

    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        break

    # Stream run for this input and thread. The checkpointer (InMemorySaver) will
    # automatically persist and restore the state keyed by thread_id between calls.
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )

    # Print messages as they arrive from the graph
    for event in events:
        if "messages" in event:
            try:
                event["messages"][-1].pretty_print()
            except Exception:
                msg = event["messages"][-1]
                if hasattr(msg, "content"):
                    print("Assistant:", msg.content)
                elif isinstance(msg, dict) and "content" in msg:
                    print("Assistant:", msg["content"])
                else:
                    print("Assistant (raw):", repr(msg))
