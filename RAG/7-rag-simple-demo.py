# 7-rag-simple-demo.py
# pip install -U python-dotenv langchain langchain-community langchain-openai \
#                 langchain-text-splitters pypdf faiss-cpu

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders.parsers.pdf import PyPDFParser
from langchain_core.documents.base import Blob
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


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


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(
        """You are a helpful assistant. Answer the question **using only** the context.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}"""
    )


def main():
    load_dotenv()  # expects OPENAI_API_KEY in .env

    # ----- Config -----
    pdf_path = "./Arjun_Varma_Generative_AI_Resume.pdf"
    embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")  # change if you like
    k = int(os.getenv("RAG_TOP_K", "2"))
    index_dir = "faiss_index"
    # -------------------

    # 1) Parse + chunk
    docs = load_pdf_docs(pdf_path)
    chunks = split_docs(docs, chunk_size=300, chunk_overlap=60)
    print(f"Chunks: {len(chunks)}")

    # 2) Embeddings + FAISS
    embeddings = OpenAIEmbeddings(model=embed_model)
    vector_store = FAISS.from_documents(chunks, embedding=embeddings)
    vector_store.save_local(index_dir)
    print(f"Saved FAISS index -> {index_dir}/")

    # 3) Retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    # 4) User question
    query = "What is the experience of Arjun Varma in AI?"

    # Retrieve most relevant chunks
    docs = retriever.invoke(query)
    print("\nRetrieved documents:")
    for d in docs:
        print(d.page_content[:200].strip(), "…\n", "-" * 60)

    # 5) Prompt + LLM chain
    prompt = build_prompt()
    llm = ChatOpenAI(model=chat_model)
    chain = prompt | llm | StrOutputParser()

    # 6) Run
    context = "\n\n".join(d.page_content for d in docs)
    user_input = {"context": context, "question": query}
    response = chain.invoke(user_input)

    print("\nResponse:\n", response)


if __name__ == "__main__":
    main()
