# 13_rag_one_off_questions.py
"""
One-off RAG question over a persisted Chroma DB.

- Loads ./db/chroma_db (configurable via env)
- Uses the same embedding model used to build the DB
- Retrieves top-k chunks for a single query (CLI arg or default)
- Sends ONLY retrieved content to the LLM
- Prints the model's answer (result.content)

Usage:
  python 13_rag_one_off_questions.py "What does Dracula fear the most?"
"""

import os
import sys
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


def get_current_dir() -> str:
    """Safe directory detection for scripts/notebooks."""
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


def main():
    load_dotenv()  # expects OPENAI_API_KEY in .env or env

    # --- Config (can override via environment variables) ---
    current_dir = get_current_dir()
    db_root = os.environ.get("RAG_DB_DIR", os.path.join(current_dir, "db"))
    chroma_dir = os.environ.get("RAG_CHROMA_DIR", "chroma_db")  # folder created earlier
    persist_dir = os.path.join(db_root, chroma_dir)

    embed_model = os.environ.get("RAG_EMBED_MODEL", "text-embedding-3-small")
    chat_model = os.environ.get("RAG_CHAT_MODEL", "gpt-4o-mini")  # or gpt-4o
    top_k = int(os.environ.get("RAG_TOP_K", "3"))

    # Query from CLI or default:
    query = "What does Dracula fear the most?"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])

    # --- Load Chroma with the same embedding function used to build it ---
    embeddings = OpenAIEmbeddings(model=embed_model)
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    # --- Build retriever & fetch relevant docs ---
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    relevant_docs = retriever.invoke(query)

    # Optional: show what we retrieved
    print("\n--- Relevant Documents ---")
    if not relevant_docs:
        print("No results. Add more data or increase RAG_TOP_K.")
    for i, d in enumerate(relevant_docs, 1):
        print(f"\nDocument {i}:\n{d.page_content[:400]}...")
        print(f"Source: {(d.metadata or {}).get('source', 'unknown')}")

    # --- Combine retrieved chunks into a single prompt for the LLM ---
    combined_input = (
        "Here are some documents that might help answer the question.\n"
        f"Question: {query}\n\n"
        "Relevant Documents:\n\n"
        + "\n\n".join(doc.page_content for doc in relevant_docs)
        + "\n\nPlease answer ONLY using the content above. "
          "If the answer is not present, reply: \"I'm not sure.\""
    )

    # --- Call the chat model ---
    model = ChatOpenAI(model=chat_model, temperature=0.0)
    messages = [
        SystemMessage(content="You are a concise, helpful assistant."),
        HumanMessage(content=combined_input),
    ]
    result = model.invoke(messages)

    # --- Print just the answer text ---
    print("\n--- Generated Response ---")
    print(result.content)


if __name__ == "__main__":
    main()
