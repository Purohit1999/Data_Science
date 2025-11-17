 # 10_b_basic_part_2.py
"""
Load a persisted Chroma vector store and run retrieval with a similarity score threshold.
Requires the DB created in part 1 (persisted to ./chroma_db by default).
"""

import os
import sys
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


def main():
    load_dotenv()  # loads OPENAI_API_KEY if present

    # --- Paths ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    persistent_directory = os.path.join(
        current_dir, os.environ.get("RAG_CHROMA_DIR", "chroma_db")
    )

    if not os.path.exists(persistent_directory):
        raise FileNotFoundError(
            f"Chroma directory not found: {persistent_directory}\n"
            "Run the builder script (part 1) to create it first."
        )

    # --- Embeddings + Chroma (must pass embedding_function when loading) ---
    embeddings = OpenAIEmbeddings(model=os.environ.get("RAG_EMBED_MODEL", "text-embedding-3-small"))
    db = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embeddings,
    )

    # --- Query (CLI arg or default) ---
    query = "Where does Gandalf meet Frodo?"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])

    # --- Retriever with threshold ---
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": int(os.environ.get("RAG_TOP_K", "10")),
            "score_threshold": float(os.environ.get("RAG_SCORE_THRESHOLD", "0.5")),
        },
    )

    # --- Retrieve ---
    relevant_docs = retriever.invoke(query)

    # --- Display ---
    print("\n--- Relevant Documents ---")
    if not relevant_docs:
        print("No results met the similarity threshold. Try lowering RAG_SCORE_THRESHOLD.")
        return

    for i, doc in enumerate(relevant_docs, 1):
        print(f"\nDocument {i}:")
        print(doc.page_content[:1000])  # truncate for neat output
        src = (doc.metadata or {}).get("source", "unknown")
        print(f"Source: {src}")
        if "score" in (doc.metadata or {}):
            print(f"Score: {doc.metadata['score']}")

    print("\nQuery:", query)


if __name__ == "__main__":
    main()
