# 11_a_rag_basic_metadata.py
"""
Create & use a Chroma vector DB for MULTIPLE text files with metadata.

- Loads all .txt files from ./books (configurable)
- Adds metadata: {"source": <filename>, "path": <absolute_path>}
- Splits into chunks
- Embeds with OpenAI
- Persists to ./db/chroma_db
- Example retrieval prints chunks + sources
"""

import os
from glob import glob
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


def build_or_load_db(
    books_dir: str,
    persist_dir: str,
    embed_model: str = "text-embedding-3-small",
    chunk_size: int = 1000,
    chunk_overlap: int = 50,
) -> Chroma:
    """Create the Chroma DB if missing; otherwise load it."""
    os.makedirs(os.path.dirname(persist_dir), exist_ok=True)

    embeddings = OpenAIEmbeddings(model=embed_model)

    # If DB already exists, just open and return it
    if os.path.exists(persist_dir):
        print("✅ Vector store already exists. Loading…")
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    # Otherwise, build it from the text files
    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"Books directory not found: {books_dir}\n"
            "Create it and drop your .txt files inside."
        )

    files = sorted(glob(os.path.join(books_dir, "*.txt")))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {books_dir}")

    print(f"📚 Found {len(files)} file(s):")
    for f in files:
        print(" -", os.path.basename(f))

    # Load all files into Documents and attach metadata
    documents = []
    for path in files:
        loader = TextLoader(path, encoding="utf-8")
        file_docs = loader.load()
        for d in file_docs:
            d.metadata = {
                "source": os.path.basename(path),
                "path": os.path.abspath(path),
            }
            documents.append(d)

    # Split into chunks
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)

    print(f"\n🧩 Total chunks created: {len(chunks)}")
    if chunks:
        print("Sample chunk:\n", chunks[0].page_content[:500], "\n")

    # Create and persist Chroma DB
    db = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
    db.persist()
    print(f"💾 Vector store created & persisted at: {persist_dir}")
    return db


def main():
    load_dotenv()  # loads OPENAI_API_KEY, etc.

    # --- Config (env-overridable) ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    books_dir = os.path.join(current_dir, os.environ.get("RAG_BOOKS_DIR", "books"))
    db_root = os.path.join(current_dir, os.environ.get("RAG_DB_DIR", "db"))
    persist_dir = os.path.join(db_root, os.environ.get("RAG_CHROMA_DIR", "chroma_db"))

    embed_model = os.environ.get("RAG_EMBED_MODEL", "text-embedding-3-small")
    chunk_size = int(os.environ.get("RAG_CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.environ.get("RAG_CHUNK_OVERLAP", "50"))

    # Build or load DB
    db = build_or_load_db(
        books_dir=books_dir,
        persist_dir=persist_dir,
        embed_model=embed_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # --- Retrieval test ---
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    query = "Where does Gandalf meet Frodo?"
    results = retriever.invoke(query)

    print("\n🔎 Query:", query)
    print("\n--- Query Results ---")
    if not results:
        print("No results. Try adjusting k or adding more files.")
        return

    for i, doc in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(doc.page_content[:1000])  # truncate for neat output
        meta = doc.metadata or {}
        print(f"Source: {meta.get('source', 'unknown')}")
        print(f"Path:   {meta.get('path', 'unknown')}")


if __name__ == "__main__":
    main()
