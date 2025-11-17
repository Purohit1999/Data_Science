# 12b_rag_basics_metadata.py
"""
This program:
✅ Loads an existing Chroma vector database (created in part 11)
✅ Retrieves the most relevant text chunks based on a user query
✅ Combines those chunks into a single prompt
✅ Sends the prompt to OpenAI (ChatGPT) using ChatOpenAI
✅ Prints the model’s generated answer

It demonstrates the second stage of a RAG (Retrieval-Augmented Generation) pipeline.
"""

# --- Imports ---
import os
from dotenv import load_dotenv

# LangChain + Chroma imports
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


# --- Load API key and environment variables ---
load_dotenv()  # Reads from your local .env file (OPENAI_API_KEY=sk-...)


# --- Step 1: Safe path handling (works in both scripts & notebooks) ---
try:
    # __file__ gives current file path; works when running .py file
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # __file__ may not exist inside Jupyter or REPL
    current_dir = os.getcwd()

# Define where your Chroma DB is stored (adjust folder names if needed)
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

# Ensure the directory exists (creates it if missing)
os.makedirs(persistent_directory, exist_ok=True)


# --- Step 2: Define embeddings and load the Chroma DB ---
# The same embedding model must be used that created the DB
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load existing Chroma DB from disk
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings,
)


# --- Step 3: Build retriever for similarity search ---
# Retrieves documents that are semantically similar to your query
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.2},
)

# Define the user query (the question to be answered)
query = "Where is Dracula's castle located?"


# --- Step 4: Retrieve relevant documents ---
# Depending on your LangChain version, use get_relevant_documents() or retrieve()
try:
    relevant_docs = retriever.get_relevant_documents(query)
except AttributeError:
    # Some older versions of LangChain call this method 'retrieve'
    relevant_docs = retriever.retrieve(query)


# --- Step 5: Print retrieved document chunks and metadata ---
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, start=1):
    # Safely get text and metadata from each document
    content = getattr(doc, "page_content", str(doc))
    src = doc.metadata.get("source") if hasattr(doc, "metadata") else None
    print(f"\nDocument {i}:\n{content[:400]}...")  # Print first 400 chars for brevity
    print(f"Source: {src}\n")


# --- Step 6: Combine all retrieved chunks into a single context string ---
# This is the "R" (retrieval) part that feeds the LLM context
combined_input = (
    f"Here are some documents that might help answer the question:\n\n"
    f"Question: {query}\n\n"
    f"Relevant Documents:\n"
    + "\n\n".join([getattr(d, "page_content", "") for d in relevant_docs])
    + "\n\nPlease provide a clear, concise answer based only on these documents. "
    + "If the answer isn't found in the text, reply: 'I'm not sure.'"
)


# --- Step 7: Call ChatOpenAI (the "G" in RAG) ---
# Use predict_messages() if available, otherwise fallback to model()
model = ChatOpenAI(model="gpt-4o")  # You can change to "gpt-4o-mini" or "gpt-3.5-turbo"

# Prepare chat-style input for LLM
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]


# --- Step 8: Try different model methods (depending on LangChain version) ---
try:
    # Most recent LangChain supports predict_messages()
    response = model.predict_messages(messages)
    assistant_text = response.content
except AttributeError:
    # Older versions may only support model(messages)
    try:
        resp = model(messages)
        assistant_text = getattr(resp, "content", str(resp))
    except Exception:
        # As last fallback, use generate()
        gen = model.generate([messages])
        assistant_text = gen.generations[0][0].text


# --- Step 9: Display final LLM response ---
print("\n--- Generated Response ---")
print(assistant_text)
