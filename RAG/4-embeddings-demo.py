# 4-embeddings-demo.py
# pip install -U python-dotenv langchain langchain-openai langchain-google-genai

from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv

# LangChain embedding wrappers
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def choose_embedder() -> tuple[str, object]:
    """
    Choose an embedding model based on available API keys.
    Priority: OpenAI -> Google.
    Returns (provider_name, embeddings_instance).
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")

    if openai_key:
        # Good general-purpose, 1536 dims
        model_name = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        emb = OpenAIEmbeddings(model=model_name)
        return ("openai", emb)

    if google_key:
        # Google’s current embedding model
        model_name = os.getenv("GOOGLE_EMBED_MODEL", "text-embedding-004")
        emb = GoogleGenerativeAIEmbeddings(model=model_name)
        return ("google", emb)

    raise RuntimeError(
        "No embedding provider configured. "
        "Set OPENAI_API_KEY or GOOGLE_API_KEY in your environment/.env."
    )


def pretty_preview(v: List[float], n: int = 10) -> str:
    head = ", ".join(f"{x:.4f}" for x in v[:n])
    return f"[{head}{', ...' if len(v) > n else ''}]"


def main():
    load_dotenv()  # load variables from .env

    provider, model = choose_embedder()
    print(f"Using provider: {provider}")
    print(f"Embedding model: {getattr(model, 'model', 'unknown')}")

    # ----- Example texts to embed -----
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is the simulation of human intelligence in machines.",
        "LangChain is a framework for building applications with LLMs.",
        "Embeddings capture semantic meaning of text as vectors.",
    ]

    # ----- Generate embeddings for the batch of texts -----
    print("\nCreating document embeddings...")
    embeddings = model.embed_documents(texts)  # List[List[float]]

    print(f"Created {len(embeddings)} embeddings.\n")
    for i, emb in enumerate(embeddings, start=1):
        print(
            f"Text {i}: dim={len(emb)}  preview={pretty_preview(emb, 10)}"
        )

    # ----- Single-query embedding example -----
    query = "What is artificial intelligence?"
    q_vec = model.embed_query(query)
    print(f"\nQuery embedding: dim={len(q_vec)}  preview={pretty_preview(q_vec, 10)}")


if __name__ == "__main__":
    main()
