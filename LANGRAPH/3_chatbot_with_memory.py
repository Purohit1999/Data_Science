"""
langgraph_memory_chatbot.py

A LangGraph + LangChain example that:
 - Uses an OpenAI chat model (gpt-4o-nano) via langchain's init_chat_model
 - Attaches a Tavily web-search tool
 - Builds a LangGraph StateGraph with a chatbot node and a ToolNode
 - Uses InMemorySaver as a simple checkpointer to persist conversation threads (memory)
 - Streams responses and prints the assistant's message with a simple pretty_print call

Required environment variables (in your .env):
 - OPENAI_API_KEY
 - TAVILY_API_KEY

Run:
  python langgraph_memory_chatbot.py
"""

# -------------------------
# Imports & environment
# -------------------------
from typing import Annotated
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model   # helper to init chat LLMs (LangChain)
from langchain_tavily import TavilySearch         # Tavily search wrapper for LangChain
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv

# Load environment variables (API keys) from .env into the process environment.
# This enables libraries like langchain to read OPENAI_API_KEY and TAVILY_API_KEY.
load_dotenv()

# -------------------------
# 1. State schema
# -------------------------
class State(TypedDict):
    """
    TypedDict schema for the state that flows through the graph.

    Fields:
      - messages: Annotated list of messages. Each message is typically a mapping like:
            {"role": "user"|"assistant"|"system", "content": "<text>"}

    The `Annotated[..., add_messages]` part tells LangGraph to append returned messages
    to the existing `messages` list instead of replacing it.
    """
    messages: Annotated[list, add_messages]


# -------------------------
# 2. Create the graph builder
# -------------------------
# StateGraph(State) creates a builder that will validate/operate on states matching the schema
graph_builder = StateGraph(State)


# -------------------------
# 3. Initialize model + tools
# -------------------------
# init_chat_model("openai:gpt-4o-nano") creates a chat model object using LangChain's wrapper
# (this wrapper will use the OPENAI_API_KEY from environment variables).
llm = init_chat_model("openai:gpt-5-nano")

# TavilySearch is a web-search tool that the LLM can call. max_results controls how many
# top search results to return when the tool is invoked.
tool = TavilySearch(max_results=2)

# If you have multiple tools, put them in this list.
tools = [tool]

# bind_tools attaches the tools to the LLM wrapper. The returned object (llm_with_tools)
# behaves like a chat model but the model may produce "tool call" actions which the wrapper
# will execute (i.e., call Tavily when the model asks for it).
llm_with_tools = llm.bind_tools(tools)


# -------------------------
# 4. Chatbot node function
# -------------------------
def chatbot(state: State) -> dict:
    """
    Node function that implements the chatbot behavior.

    Input:
      - state: The current state (must follow the State TypedDict shape).
               Example: {"messages": [{"role": "user", "content": "Hi"}]}

    What it does:
      - Calls the LLM (with tools bound) using the provided chat history (state["messages"])
      - Receives an assistant message (usually an object or dict containing the generated text)
      - Returns a dict mapping state keys to new values. Returning {"messages": [assistant_msg]}
        tells LangGraph to append the assistant_msg to the state.messages because we used add_messages.

    Return:
      - A dictionary of updates to the State. LangGraph merges this update into the running state.
    """
    # model.invoke(...) sends the message history to the LLM and returns an assistant message.
    assistant_message = llm_with_tools.invoke(state["messages"])

    # Return messages list — add_messages will append the assistant message to the state's messages.
    return {"messages": [assistant_message]}


# Register the chatbot function as a node in the builder.
graph_builder.add_node("chatbot", chatbot)


# -------------------------
# 5. Tool node & conditional routing
# -------------------------
# ToolNode is a prebuilt node that knows how to run the configured tools and attach
# their results back into the graph state. It provides a standard way to integrate
# "external" calls like web search into the graph flow.
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# tools_condition is a helper that inspects the LLM output and decides whether
# external tools should be called. It wires up conditional edges from "chatbot" to "tools".
# Example effect: if model output indicates it needs web info, the flow will go to tools.
graph_builder.add_conditional_edges("chatbot", tools_condition)

# After tools run, route back to chatbot so LLM can incorporate tool results into its reply.
graph_builder.add_edge("tools", "chatbot")

# Set the entry point to "chatbot" (which node to start the graph from).
graph_builder.set_entry_point("chatbot")


# -------------------------
# 6. Checkpoint / Memory (InMemorySaver)
# -------------------------
# InMemorySaver is a simple checkpointer implementation that stores conversation state
# in memory. It's useful for demos or single-process runs. For production you would use
# a persistent checkpointer (database, file, or external store).
memory = InMemorySaver()

# Compile the graph into a runnable Graph instance.
# Passing checkpointer=memory tells the compiled graph to use this saver to persist
# states (for example keyed by thread_id).
# compile() validates the graph, constructs runtime structures, and returns an object with
# methods like .stream() and .get_graph().
graph = graph_builder.compile(checkpointer=memory)


# -------------------------
# 7. Save graph visualization
# -------------------------
# draw_mermaid_png() returns PNG bytes rendering the graph as a Mermaid diagram.
# We persist it locally so you can open it in an image viewer.
png_data = graph.get_graph().draw_mermaid_png()
with open("3_chatbot_with_memory.png", "wb") as f:
    f.write(png_data)
print("✅ Graph saved as 3_chatbot_with_memory.png")


# -------------------------
# 8. Interactive loop (main usage)
# -------------------------
# This loop allows multiple conversations (threads). The code asks for a thread_id
# that will be used as a key by the checkpointer to store and retrieve conversation state.
# This enables memory across runs within the same process using InMemorySaver.
while True:
    # Ask the user for a thread_id. This lets you simulate separate conversation sessions.
    # Any string works (numbers, names, UUIDs). Example: "1", "support_ticket_42", "session-A".
    thread_id = input("Enter a thread id for this conversation (any string, e.g., '1'): ")
    # Build the config object that the LangGraph runtime expects. The exact shape can vary
    # by version; here we provide a `configurable` dict with thread_id.
    # The compiled graph will pass this to the checkpointer so it can load/save per-thread state.
    config = {"configurable": {"thread_id": thread_id}}

    print("\n🤖 Chatbot with Memory is ready! Type 'quit', 'exit', or 'q' to stop.\n")

    # Get user input for the current thread
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("👋 Goodbye!")
        break

    # Graph.stream: starts the graph with provided initial state and yields events.
    # We pass:
    #   - initial state: {"messages": [{"role": "user", "content": user_input}]}
    #   - config: includes the thread_id so memory can be loaded/saved
    #   - stream_mode="values": instructs the runtime to yield the actual state values
    #                        (instead of full event envelopes or other formats).
    #
    # The returned `events` is an iterator/generator. Each element is typically a state-like dict
    # or an event that contains the updated state for a node. Depending on LangGraph's internal
    # version, the exact structure may vary; here we assume each `event` is a plain state dict.
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )

    # Iterate through events produced while the graph runs. Usually the final event includes
    # the assistant's reply appended to the messages list.
    for event in events:
        # The event is expected to be a state-like dict: {"messages": [..., assistant_msg]}
        # We access the last message:
        last_msg = event["messages"][-1]

        # pretty_print() is a convenience method provided by some message wrappers to print
        # the message in a readable way (role + content, maybe metadata). If the object
        # doesn't support pretty_print, you could print .content or str(...) instead.
        #
        # We call it defensively; in real code you'd check capabilities or normalize message type.
        try:
            # Many LangChain message wrappers provide pretty_print
            last_msg.pretty_print()
        except Exception:
            # Fallback printing for plain dicts or simple objects
            if hasattr(last_msg, "content"):
                print("Assistant:", last_msg.content)
            elif isinstance(last_msg, dict) and "content" in last_msg:
                print("Assistant:", last_msg["content"])
            else:
                print("Assistant (raw):", repr(last_msg))

# End of script
