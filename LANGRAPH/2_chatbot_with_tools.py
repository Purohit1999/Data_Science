"""
2_chatbot_with_tools.py

LangGraph + LangChain example that:

- Uses an OpenAI chat model (gpt-5-nano) via langchain's init_chat_model
- Attaches a Tavily web-search tool
- Builds a LangGraph StateGraph with a chatbot node and a ToolNode
- Streams responses from the graph to the terminal
- Saves a PNG visualization of the graph

Required environment variables (in your .env or shell):

- OPENAI_API_KEY
- TAVILY_API_KEY

Run:

    python 2_chatbot_with_tools.py
"""

# ================================
# Imports & environment
# ================================
import os
from typing import Annotated, Dict, Any

from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# Load env vars from .env (if present)
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not set in environment or .env file")

if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("TAVILY_API_KEY not set in environment or .env file")


# ================================
# 1. Define the shape of state
# ================================
class State(TypedDict):
    """
    TypedDict schema for the state that flows through the graph.

    - messages: list of chat messages
      Each message is usually a dict like:
          {"role": "user" | "assistant" | "system", "content": "..."}

    We annotate it with `add_messages` so that LangGraph knows to
    APPEND new messages instead of replacing the list.
    """

    messages: Annotated[list, add_messages]


# ================================
# 2. Create a graph builder
# ================================
# StateGraph(State) creates a builder that accepts/validates states
# following the State TypedDict above.
graph_builder = StateGraph(State)


# ================================
# 3. Initialize model and tools
# ================================

# init_chat_model is a convenience wrapper that returns a chat model.
# "openai:gpt-5-nano" uses OpenAI's gpt-5-nano model via the OpenAI provider.
model = init_chat_model("openai:gpt-5-nano")

# TavilySearch is the LangChain wrapper that allows the LLM to perform web searches.
# max_results=2 instructs the tool to return up to 2 search results per call.
tool = TavilySearch(max_results=2)

# If you have multiple tools, put them in this list.
tools = [tool]

# bind_tools attaches the tool wrappers to the model so the model can decide
# when to call them. The returned object behaves like a chat model.
llm_with_tools = model.bind_tools(tools)


# ================================
# 4. Define the chatbot node function
# ================================
def chatbot(state: State) -> Dict[str, Any]:
    """
    Chatbot node function for the graph.

    Inputs:
    - state: current State dict. Must match the State TypedDict shape.

    Behaviour:
    - Calls the LLM (with tools bound) using the existing messages.
    - Returns an update of the form {"messages": [assistant_message]},
      which will be appended to the state's messages list (because of
      the add_messages annotation).

    Returns:
    - dict mapping keys in State to values to update.
    """
    # llm_with_tools.invoke expects a list of messages (chat history).
    assistant_message = llm_with_tools.invoke(state["messages"])

    return {"messages": [assistant_message]}


# ================================
# 5. Add chatbot node to the builder
# ================================
# Registers a node named "chatbot" that runs chatbot(state).
graph_builder.add_node("chatbot", chatbot)


# ================================
# 6. Create and add ToolNode
# ================================
# ToolNode is a prebuilt node that knows how to call tool wrappers
# (like TavilySearch), format their I/O, and merge results back into state.
tool_node = ToolNode(tools=tools)

# Add the tool node under the name "tools".
graph_builder.add_node("tools", tool_node)


# ================================
# 7. Connect nodes with edges
# ================================
# Conditional edges from "chatbot" using tools_condition:
# - tools_condition inspects the LLM output and decides if a tool call is needed.
graph_builder.add_conditional_edges("chatbot", tools_condition)

# After tools run, route back to chatbot so the model can integrate tool results.
graph_builder.add_edge("tools", "chatbot")

# Start the graph at "chatbot".
graph_builder.add_edge(START, "chatbot")

# Optionally allow chatbot -> END (explicit termination).
graph_builder.add_edge("chatbot", END)


# ================================
# 8. Compile graph
# ================================
# compile() validates the graph (start/end, connectivity, etc.)
# and returns a runnable Graph instance with .invoke() and .stream().
graph = graph_builder.compile()


# ================================
# 9. Save graph visualization (optional)
# ================================
# draw_mermaid_png() returns PNG bytes of a Mermaid diagram.
try:
    png_data = graph.get_graph().draw_mermaid_png()
    out_file = "2_chatbot_with_tools.png"
    with open(out_file, "wb") as f:
        f.write(png_data)
    print(f"Graph saved as {out_file}")
except Exception as e:
    # Visualization requires extra deps; if it fails, that's fine.
    print(f"Could not generate Mermaid PNG (ok to ignore): {e}")


# ================================
# 10. Streaming helper function
# ================================
def stream_graph_updates(user_input: str) -> None:
    """
    Stream updates from the compiled graph for a single user input.

    How it works:
    - Wraps the user message into an initial State with one "user" message.
    - Calls graph.stream(initial_state), which yields events
      as nodes produce partial or full state updates.
    - For each event, it prints the **latest assistant message**.

    Parameters:
    - user_input: text typed by the user.


Notes:
 - Each event returned by graph.stream() is typically a mapping where
   the values are partial state dicts. We access the last message with
   value["messages"][-1].

 - Use this function when you want to stream intermediate outputs while
   the LLM/tool run completes (e.g., streaming tokens or multiple node
   outputs).


    """

    # Prepare initial state; add_messages ensures this appears in history.
    initial_state: State = {
        "messages": [{"role": "user", "content": user_input}]
    }

    # graph.stream returns an iterator of events (partial state dicts).
    for event in graph.stream(initial_state, stream_mode="values"):
        # Each event may contain multiple values (e.g., for different branches).
        for value in event.values():
            if not isinstance(value, dict) or "messages" not in value:
                continue

            # Take the last message as the most recent assistant reply.
            assistant_msg = value["messages"][-1]

            # Different wrappers may return slightly different objects; handle both.
            if hasattr(assistant_msg, "content"):
                # LangChain-style Message object
                print("Assistant:", assistant_msg.content)
            elif isinstance(assistant_msg, dict) and "content" in assistant_msg:
                # Plain dict
                print("Assistant:", assistant_msg["content"])
            else:
                # Fallback: print the raw object
                print("Assistant:", repr(assistant_msg))


# ================================
# 11. Main interactive loop (REPL)
# ================================
def main_loop() -> None:
    """
    Simple REPL (read–eval–print loop) for the chatbot.

    - Reads user input from stdin.
    - Streams the graph's response via stream_graph_updates().
    - Exits cleanly when the user types 'quit', 'exit', or 'q'.
    """

    print("Chatbot is ready! Type 'quit', 'exit', or 'q' to stop.\n")

    while True:
        try:
            # Ask for user input
            user_input = input("User: ")

            # Exit commands
            if user_input.strip().lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            # Stream responses for this user input
            stream_graph_updates(user_input)

        except KeyboardInterrupt:
            # Graceful exit on Ctrl+C
            print("\nInterrupted. Goodbye!")
            break
        except Exception as e:
            # Helpful for debugging runtime issues (bad env vars, etc.)
            print("Error:", e)
            break


# ================================
# Entry point
# ================================
if __name__ == "__main__":
    main_loop()
