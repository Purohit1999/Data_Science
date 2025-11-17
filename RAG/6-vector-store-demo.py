# 6-vector-store-demo.py
# pip install -U python-dotenv langchain langchain-community langchain-openai \
#                 langchain-text-splitters pypdf faiss-cpu

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders.parsers.pdf import PyPDFParser
from langchain_core.documents.base import Blob
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def load_pdf_docs(pdf_path: str):
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path.resolve()}")
    blob = Blob.from_path(str(path))
    parser = PyPDFParser()
    return [d for d in parser.lazy_parse(blob)]


def split_docs(docs, chunk_size=300, chunk_overlap=60):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    return splitter.split_documents(docs)


def main():
    load_dotenv()  # needs OPENAI_API_KEY in .env

    # ---- change to your PDF path ----
    pdf_path = "./Arjun_Varma_Generative_AI_Resume.pdf"
    # ---------------------------------

    print(f"Parsing: {pdf_path}")
    docs = load_pdf_docs(pdf_path)

    print("Splitting into chunks...")
    chunks = split_docs(docs, chunk_size=300, chunk_overlap=60)
    print(f"Number of chunks created: {len(chunks)}")

    # Embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Create FAISS vector store (no external DB required)
    print("Building FAISS index...")
    vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)

    # Optional: save locally for reuse
    index_dir = "faiss_index"
    vector_store.save_local(index_dir)
    print(f"Saved FAISS index to: {index_dir}/")

    # ---- Query the store ----
    query = "AI engineer"
    print(f"\nResults for query: {query!r}")
    results = vector_store.similarity_search(query, k=3)
    for i, r in enumerate(results, start=1):
        print(f"\nResult {i}:")
        print("Content:", r.page_content)
        print("Metadata:", r.metadata)

    # ---- Reload example (optional) ----
    # Note: allow_dangerous_deserialization is required for FAISS local load.
    print("\nReloading FAISS index from disk for verification…")
    reloaded = FAISS.load_local(
        index_dir, embeddings=embeddings, allow_dangerous_deserialization=True
    )
    again = reloaded.similarity_search(query, k=1)
    print("\nReloaded top match snippet:", again[0].page_content[:200], "…")


if __name__ == "__main__":
    main()
