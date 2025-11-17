# Importing TypedDict for defining a structured dictionary type
from typing import TypedDict

# Define a TypedDict to represent the state of a portfolio
# This helps with type checking and ensures our dictionary has these specific keys
class PortfolioState(TypedDict):
    amount_usd: float   # The initial amount in USD
    total_usd: float    # Total USD after calculation (e.g., adding interest)
    total_inr: float    # Total INR after conversion from USD


# Function to calculate the total USD
def calc_total(state: PortfolioState) -> PortfolioState:
    # Example: add 8% interest to amount_usd
    state["total_usd"] = state["amount_usd"] * 1.08
    return state


# Function to convert USD to INR
def convert_to_inr(state: PortfolioState) -> PortfolioState:
    # Convert total_usd to INR using exchange rate (e.g., 1 USD = 85 INR)
    state["total_inr"] = state["total_usd"] * 85
    return state


# --- Conditional function ---
def check_conversion(state: PortfolioState) -> str:
    # If total_usd is more than 2000, convert to INR
    if state["total_usd"] > 2000:
        return "Convert_to_inr_node"
    else:
        return END  # directly end if not enough USD


# Importing LangGraph to build a state graph workflow
from langgraph.graph import StateGraph, START, END

# Create a state graph builder with PortfolioState as the state type
builder = StateGraph(PortfolioState)

# Add nodes (functions) to the graph
builder.add_node("calc_total_node", calc_total)          # Node to calculate USD total
builder.add_node("Convert_to_inr_node", convert_to_inr)  # Node to convert to INR

# Define the edges (workflow) between nodes
builder.add_edge(START, "calc_total_node")               # Start -> calc_total_node

# Instead of a fixed edge from calc_total_node -> Convert_to_inr_node,
# we use a conditional edge based on state
builder.add_conditional_edges(
    "calc_total_node",   # from this node
    check_conversion     # function decides the next node
)

builder.add_edge("Convert_to_inr_node", END)             # Convert_to_inr_node -> End

# Compile the graph so it's ready to execute
graph = builder.compile()

# Optional: visualize the graph using IPython display in Jupyter
from IPython.display import Image, display
# display(Image(graph.get_graph().draw_mermaid_png()))  # this is used for Jupyter notebook

png_graph = graph.get_graph().draw_mermaid_png()

with open("portfolio_graph.png", "wb") as f:
    f.write(png_graph)

# Invoke the graph with initial state (amount in USD)
result = graph.invoke({"amount_usd": 7000})

# Print the final state after executing all nodes
print(result)
