# ----------------------------- Imports -----------------------------
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI                  # OpenAI chat model (v0.2+)
from langchain_core.prompts import PromptTemplate        # Prompt template (v0.2+)
from langchain_core.output_parsers import StrOutputParser

# ---------------------- Load env variables ------------------------
# Ensure your .env has: OPENAI_API_KEY=sk-...
load_dotenv()

# ------------------ Initialize the OpenAI chat model --------------
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# --------------- Define a generative prompt template --------------
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful assistant that answers questions based on the provided context.\n"
        "If the context doesn't fully answer the question, generate a thoughtful, detailed "
        "response based on general knowledge.\n\n"
        "Context: {context}\n"
        "Question: {question}\n"
        "Answer:"
    ),
)

# ------------------------- Build the chain (LCEL) -----------------
# prompt -> model -> parse text
chain = prompt | model | StrOutputParser()

# ------------------------ Example user input ----------------------
user_input = {
    "context": (
        "The capital of India is New Delhi. It is the seat of the government of India and "
        "is known for its rich history and cultural heritage."
    ),
    "question": "Tell me about the capital of France and compare it briefly with New Delhi.",
}

# ------------------------- Generate response ----------------------
# Use .invoke for a single input; .batch for many; .stream for streaming.
response = chain.invoke(user_input)

# ---------------------------- Output ------------------------------
print("\n----- Output -----\n")
print(response)
