# 5-data-ingestion-demo.py
# pip install -U python-dotenv langchain langchain-community langchain-openai langchain-text-splitters pypdf

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from langchain_community.document_loaders.parsers.pdf import PyPDFParser
from langchain_core.documents.base import Blob
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings


def load_pdf_docs(pdf_path: str):
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path.resolve()}")

    blob = Blob.from_path(str(path))
    parser = PyPDFParser()
    docs = [d for d in parser.lazy_parse(blob)]
    return docs


def split_docs(docs, chunk_size: int = 100, chunk_overlap: int = 50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(docs)


def preview_text(text: str, n: int = 160) -> str:
    t = " ".join(text.split())
    return t if len(t) <= n else t[:n] + "…"


def preview_vec(vec: List[float], n: int = 10) -> str:
    head = ", ".join(f"{x:.4f}" for x in vec[:n])
    return f"[{head}{', …' if len(vec) > n else ''}]"


def main():
    load_dotenv()  # reads .env for OPENAI_API_KEY

    # ---- change this to your file ----
    pdf_path = "./Arjun_Varma_Generative_AI_Resume.pdf"
    # ----------------------------------

    print(f"Loading PDF: {pdf_path}")
    docs = load_pdf_docs(pdf_path)
    print(f"Parsed {len(docs)} document(s).")

    chunks = split_docs(docs, chunk_size=100, chunk_overlap=50)
    print(f"\nNumber of chunks created: {len(chunks)}\n")

    for i, ch in enumerate(chunks, start=1):
        print(f"Chunk {i}:")
        print(preview_text(ch.page_content))
        print("-" * 60)

    # ---- Embeddings ----
    model = OpenAIEmbeddings(model="text-embedding-3-small")
    texts = [c.page_content for c in chunks]
    print("\nCreating embeddings for chunks...")
    embeddings = model.embed_documents(texts)

    print("\nEmbeddings:")
    for i, (emb, ch) in enumerate(zip(embeddings, chunks), start=1):
        print(
            f"Chunk {i:>3} | dim={len(emb)} | text='{preview_text(ch.page_content, 60)}' "
            f"| vec={preview_vec(emb, 10)}"
        )


if __name__ == "__main__":
    main()
