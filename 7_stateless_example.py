# 7_stateless_example.py
# ---------------------------------------------------------------
# Demonstrates stateless usage of ChatOpenAI + a simple batch call
# ---------------------------------------------------------------

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Load OPENAI_API_KEY from .env (must be in the same folder)
load_dotenv()

# Initialize the OpenAI chat model (use any model available to your key)
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# ---------------------- Stateless single turns -------------------
# Turn 1: You tell the model your name
resp1 = model.invoke([
    SystemMessage(content="You are a concise, helpful assistant."),
    HumanMessage(content="My name is Nithyashree.")
])
print("\n# You: My name is Nithyashree.")
print("# AI:", resp1.content)

# Turn 2: You ask the model your name in a NEW, stateless call
resp2 = model.invoke([
    SystemMessage(content="You are a concise, helpful assistant."),
    HumanMessage(content="What is my name?")
])
print("\n# You: What is my name?")
print("# AI:", resp2.content)  # It shouldn't know—no memory between calls.

# --------------------------- Batch demo --------------------------
# Each string is treated independently (no shared state/memory).
batch_inputs = [
    "Kantharaju has won best student in my class.",
    "Who won best student in my class?"
]

batch_responses = model.batch(batch_inputs)

print("\nBatch responses:")
for i, r in enumerate(batch_responses, start=1):
    print(f"\n--- Response {i} ---")
    print(r.content)
