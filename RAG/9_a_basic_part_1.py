# 9_a_basic_part_1.py
"""
Build & persist a Chroma vector store from a local TXT file.

- Resolves paths using __file__ (portable; no hardcoded drive letters)
- Uses modern import paths for LangChain components
- Persists Chroma DB to ./chroma_db
"""

import os
from dotenv import load_dotenv

# Modern LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


def main():
    load_dotenv()  # loads OPENAI_API_KEY from .env if present

    # ---- Paths (portable, based on this script's directory) ----
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "Data")
    file_name = os.environ.get("RAG_TEXT_FILE", "lord_of_the_rings.txt")
    file_path = os.path.join(data_dir, file_name)

    persistent_directory = os.path.join(current_dir, "chroma_db")

    # ---- Sanity checks ----
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            f"Create it and put your TXT file there (e.g., {file_name})."
        )

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path or set RAG_TEXT_FILE."
        )

    # ---- If DB already exists, skip re-init (idempotent run) ----
    if not os.path.exists(persistent_directory):
        print("Persistent directory does not exist. Initializing vector store…")

        # 1) Load text file into Documents
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()

        # 2) Split into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)

        print("\n--- Document Chunks Info ---")
        print(f"Number of chunks: {len(docs)}")
        if docs:
            print(f"Sample chunk:\n{docs[0].page_content[:500]}\n")

        # 3) Create embeddings
        print("\n--- Creating embeddings ---")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        print("--- Finished creating embeddings ---")

        # 4) Create & persist Chroma vector store
        print("\n--- Creating vector store ---")
        db = Chroma.from_documents(
            docs,
            embeddings,
            persist_directory=persistent_directory,
        )
        # Persist to disk
        db.persist()
        print("--- Finished creating vector store ---")

    else:
        print("Vector store already exists. No need to initialize.")

    # ---- (Optional) Open persisted DB and test a query ----
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

    retriever = db.as_retriever(search_kwargs={"k": 2})
    query = "Who is the main protagonist?"
    results = retriever.invoke(query)

    print("\n--- Retrieval test ---")
    for i, d in enumerate(results, start=1):
        print(f"\nResult {i}:\n{d.page_content[:500]}")


if __name__ == "__main__":
    main()
