# 1_simple_chatbot.py
# --------------------------------------------------
# Simple LangGraph chatbot example
# --------------------------------------------------

from typing import Annotated, TypedDict
from typing_extensions import TypedDict as TTypedDict  # safety for some envs

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

# 1. Load environment variables (e.g. OPENAI_API_KEY from .env)
# --------------------------------------------------
load_dotenv()


# 2. Define the state schema
# --------------------------------------------------
# "messages" stores the chat history.
# Annotated with add_messages so new messages are appended instead of overwritten.
class State(TypedDict):
    messages: Annotated[list, add_messages]


# 3. Create the workflow graph builder
# --------------------------------------------------
graph_builder = StateGraph(State)  # State defines the shape of data in the graph


# 4. Initialize the chat model
# --------------------------------------------------
# Use any OpenAI-compatible model name you have access to
model = ChatOpenAI(model="gpt-4.1-mini", temperature=1.1)
# (If your course uses a different name like "gpt-4o-mini" or "gpt-5-nano",
#  just replace the model string above.)


# 5. Define chatbot node function
# --------------------------------------------------
def chatbot(state: State) -> State:
    """
    This node receives the current chat state,
    calls the LLM model to generate a response based on existing messages,
    and returns an updated state dictionary.
    """
    # model.invoke(state["messages"]) generates the assistant response
    assistant_msg = model.invoke(state["messages"])

    # Return a dictionary with updated messages
    return {"messages": [assistant_msg]}


# 6. Add nodes and edges to the graph
# --------------------------------------------------
graph_builder.add_node("chatbot", chatbot)   # Add the chatbot node
graph_builder.add_edge(START, "chatbot")     # Connect START -> chatbot
graph_builder.add_edge("chatbot", END)       # Connect chatbot -> END


# 7. Compile the workflow graph
# --------------------------------------------------
graph = graph_builder.compile()  # Finalizes the graph, making it executable


# 8. Save workflow graph as PNG (optional)
# --------------------------------------------------
try:
    png_data = graph.get_graph().draw_mermaid_png()
    with open("1_simple_chatbot.png", "wb") as f:
        f.write(png_data)
    print("Graph saved as 1_simple_chatbot.png")
except Exception as e:
    print("Could not save graph PNG:", e)


# 9. Define a function to stream chatbot responses
# --------------------------------------------------
def stream_graph_updates(user_input: str) -> None:
    """
    Streams responses from the chatbot node.
    Each new assistant message is printed as soon as it is generated.
    """
    # Initial state: one user message
    initial_state = {
        "messages": [
            {"role": "user", "content": user_input}
        ]
    }

    # stream_mode="values" returns the *state* at each step
    for value in graph.stream(initial_state, stream_mode="values"):
        # value is a State dict; we care about the latest assistant message
        last_msg = value["messages"][-1]
        print("Assistant:", last_msg.content)


# 10. Main interactive loop
# --------------------------------------------------
if __name__ == "__main__":
    print("Chatbot is ready! Type 'quit', 'exit', or 'q' to stop.\n")

    while True:
        try:
            user_input = input("User: ")

            # Exit conditions
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            # Call the streaming function with user input
            stream_graph_updates(user_input)

        except Exception as e:
            print("Error:", e)
            break
